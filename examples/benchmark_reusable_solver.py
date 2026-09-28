"""Compare warmed, synchronous repeated solves through both attachment APIs.

No AMGX builds or simulations. Initialization/setup are outside the solve timer.
Both paths use identical arrays/configuration and explicitly clear the output.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cupy as cp
from cupyx.scipy import sparse
import pyamgx
from reusable_solver import CONFIG


def measure(kind, csr, rhs, out, args):
    start_setup = perf_counter()
    with ExitStack() as cleanup:
        def own(obj):
            cleanup.callback(obj.destroy)
            return obj

        if kind == "explicit":
            cfg = own(pyamgx.Config().create_from_dict(CONFIG))
            resources = own(pyamgx.Resources().create_simple(cfg))
            A = own(pyamgx.Matrix().create(resources)).attach_CSR(csr)
            b = own(pyamgx.Vector().create(resources)).attach(rhs)
            x = own(pyamgx.Vector().create(resources)).attach(out)
            solver = own(pyamgx.Solver().create(resources, cfg))
            solver.setup(A)

            def solve():
                out.fill(0)
                solver.solve(b, x)

            def pointers():
                return (*A.attached_ptrs, b.attached_ptr, x.attached_ptr)
        else:
            solver = cleanup.enter_context(pyamgx.ReusableSolver(CONFIG))
            solver.setup(csr)

            def solve():
                out.fill(0)
                solver.solve(rhs, out=out)

            def pointers():
                ptrs = solver.attached_ptrs
                return (*ptrs["csr"], ptrs["rhs"], ptrs["solution"])

        # Include first vector attachment in startup, never in the hot-path timer.
        solve()
        cp.cuda.runtime.deviceSynchronize()
        setup_ms = (perf_counter() - start_setup) * 1000
        for _ in range(args.warmup):
            solve()
        expected = tuple(a.data.ptr for a in (csr.indptr, csr.indices, csr.data, rhs, out))
        assert pointers() == expected
        pool = cp.get_default_memory_pool()
        before = (pool.used_bytes(), pool.total_bytes())
        free_before, _ = cp.cuda.runtime.memGetInfo()
        samples = []
        for _ in range(args.batches):
            start = perf_counter()
            for _ in range(args.solves):
                solve()
            cp.cuda.runtime.deviceSynchronize()
            samples.append((perf_counter() - start) * 1e6 / args.solves)
        free_after, _ = cp.cuda.runtime.memGetInfo()
        after = (pool.used_bytes(), pool.total_bytes())
        assert pointers() == expected
        assert before == after, "CuPy allocations changed during repeated solves"
        residual = float(cp.linalg.norm(csr @ out - rhs) / cp.linalg.norm(rhs))
        error = float(cp.max(cp.abs(out - 1)))
        assert residual < 1e-8 and error < 1e-8
        return {
            "kind": kind, "startup_including_first_solve_ms": setup_ms,
            "batch_mean_us_per_solve": samples, "median_us_per_solve": median(samples),
            "iterations_last_solve": solver.iterations_number,
            "relative_residual": residual, "max_solution_error": error,
            "all_five_pointers_match": True,
            "cupy_live_bytes_before_after": [before[0], after[0]],
            "cupy_reserved_bytes_before_after": [before[1], after[1]],
            "device_memory_change_bytes": free_before - free_after,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4096])
    parser.add_argument("--solves", type=int, default=100, help="solves per timed batch")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.sizes) < 2 or min(args.solves, args.batches, args.warmup) < 1:
        parser.error("sizes must be >=2; solves, batches and warmup must be positive")
    results = []
    pyamgx.initialize()
    try:
        for n in args.sizes:
            csr = sparse.diags(
                [cp.full(n - 1, -1.), cp.full(n, 4.), cp.full(n - 1, -1.)],
                [-1, 0, 1], format="csr")
            csr.sort_indices()
            rhs, out = cp.full(n, 2.), cp.zeros(n)
            rhs[0] = rhs[-1] = 3
            # Reverse order in the second pair to reduce order/clock bias.
            phases = [measure(kind, csr, rhs, out, args)
                      for kind in ("explicit", "reusable", "reusable", "explicit")]
            times = {kind: median([sample for phase in phases if phase["kind"] == kind
                                   for sample in phase["batch_mean_us_per_solve"]])
                     for kind in ("explicit", "reusable")}
            row = {
                "n": n, "nnz": csr.nnz, "phases": phases,
                "median_us_per_solve": times,
                "reusable_minus_explicit_us": times["reusable"] - times["explicit"],
                "reusable_over_explicit": times["reusable"] / times["explicit"],
            }
            results.append(row)
            print(f"n={n}: explicit {times['explicit']:.2f} us, "
                  f"reusable {times['reusable']:.2f} us "
                  f"({row['reusable_over_explicit']:.3f}x)")
    finally:
        pyamgx.finalize()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy_version": cp.__version__, "pyamgx_extension": Path(pyamgx.__file__).name,
        "amgx_libraries": sorted({Path(line.split()[-1]).name for line in
            Path("/proc/self/maps").read_text().splitlines() if "libamgx" in line}),
        "solves_per_batch": args.solves, "batches_per_phase": args.batches,
        "warmup_solves_per_phase": args.warmup,
        "config": CONFIG, "results": results,
        "notes": ["Wall-clock latency includes synchronous GPU work, Python dispatch and CuPy output zeroing.",
                  "Both paths call out.fill(0) before every solve; no warm-start iteration advantage.",
                  "Medians summarize batch means; small differences may be timing noise.",
                  "Memory counters exclude transient peaks and allocations within native pools."],
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
