"""Pass a CuPyX CSR matrix and CuPy vectors directly to PyAMGX.

Solve tridiag(-1, 4, -1) * x = b, whose exact solution is x = 1.
Print synchronized GPU memory snapshots and optionally save exact bytes as JSON.
Run in a fresh Python process with the CSR attachment extension/library loaded.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import cupy as cp
from cupyx.scipy import sparse
import pyamgx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096, help="number of rows (default: 4096)")
    parser.add_argument("--mode", choices=("dDDI", "dFFI"), default="dDDI")
    parser.add_argument("--report", type=Path, help="write measurements and pointer checks as JSON")
    parser.add_argument(
        "--no-amgx-pool", action="store_true",
        help="disable AMGX pooling with a minimal 4 KiB initial reservation",
    )
    args = parser.parse_args()
    if not 2 <= args.n <= (2**31 - 1) // 3:
        parser.error("--n must be at least 2 and fit int32 CSR indices")
    if not hasattr(pyamgx.Matrix, "attach_CSR"):
        raise RuntimeError(f"Load the CSR attachment extension; imported {pyamgx.__file__}")

    cp.cuda.Device(0).use()  # Resources.create_simple also selects device 0.
    pool = cp.get_default_memory_pool()
    measurements = []

    def record(stage):
        cp.cuda.runtime.deviceSynchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        used = total - free
        previous = measurements[-1]["device_used_bytes"] if measurements else used
        measurements.append({
            "stage": stage,
            "device_used_bytes": used,
            "device_free_bytes": free,
            "device_total_bytes": total,
            "device_delta_previous_bytes": used - previous,
            "cupy_live_bytes": pool.used_bytes(),
            "cupy_reserved_bytes": pool.total_bytes(),
        })

    record("cuda_context_ready")
    dtype = cp.float64 if args.mode == "dDDI" else cp.float32
    n = args.n
    csr = sparse.diags(
        [cp.full(n - 1, -1, dtype=dtype), cp.full(n, 4, dtype=dtype),
         cp.full(n - 1, -1, dtype=dtype)],
        [-1, 0, 1], shape=(n, n), format="csr",
    )
    csr.sort_indices()  # Prepare canonical CSR before borrowing its buffers.
    rhs = cp.full(n, 2, dtype=dtype)
    rhs[0] = rhs[-1] = 3
    solution = cp.zeros(n, dtype=dtype)
    record("cupy_inputs_ready")

    names = ("indptr", "indices", "data", "rhs", "solution")
    payload = {
        name: {"bytes": array.nbytes, "cupy_ptr": array.data.ptr}
        for name, array in zip(names, (csr.indptr, csr.indices, csr.data, rhs, solution))
    }
    expected_ptrs = tuple(payload[name]["cupy_ptr"] for name in names)
    pointer_checks = {}
    config = {
        "config_version": 2, "exception_handling": 0,
        "solver(main)": "FGMRES", "main:preconditioner": "NOSOLVER",
        "main:max_iters": 100, "main:gmres_n_restart": 20,
        "main:tolerance": 1e-10 if args.mode == "dDDI" else 1e-6,
        "main:monitor_residual": 1, "main:convergence": "RELATIVE_INI_CORE",
    }
    if args.no_amgx_pool:
        # The flag alone does not prevent the initial reservation, and this
        # AMGX pool constructor rejects zero bytes. Keep one 4 KiB page.
        config.update(device_mem_pool_enabled=0, device_mem_pool_size=4096)

    pyamgx.initialize()
    try:
        record("amgx_initialized")
        with ExitStack() as cleanup:
            def own(obj):
                cleanup.callback(obj.destroy)
                return obj

            # AMGX's text parser accepts the size_t pool-size setting; its JSON
            # parser rejects that setting as an int/size_t type mismatch.
            cfg = own(pyamgx.Config().create(",".join(f"{key}={value}" for key, value in config.items())))
            resources = own(pyamgx.Resources().create_simple(cfg))
            record("amgx_resources_ready")
            A = own(pyamgx.Matrix().create(resources, args.mode))
            b = own(pyamgx.Vector().create(resources, args.mode))
            x = own(pyamgx.Vector().create(resources, args.mode))
            solver = own(pyamgx.Solver().create(resources, cfg, args.mode))
            record("empty_handles_ready")

            # These calls receive the CuPyX/CuPy objects themselves.
            A.attach_CSR(csr)
            record("matrix_attached")
            b.attach(rhs)
            x.attach(solution)
            record("vectors_attached")

            def check_pointers(stage):
                actual = (*A.attached_ptrs, b.attached_ptr, x.attached_ptr)
                if actual != expected_ptrs:
                    raise AssertionError(f"shared buffer pointers changed at {stage}")
                pointer_checks[stage] = dict(zip(names, actual))

            check_pointers("attached")
            solver.setup(A)
            record("solver_setup")
            check_pointers("after_setup")
            solver.solve(b, x)
            record("solver_solve")
            check_pointers("after_solve")

            # AMGX has already written the answer into the original CuPy array.
            # Only diagnostic scalars are brought to the CPU.
            relative_residual = float(cp.linalg.norm(csr @ solution - rhs) / cp.linalg.norm(rhs))
            max_solution_error = float(cp.max(cp.abs(solution - 1)))
            limit = 1e-8 if args.mode == "dDDI" else 2e-5
            if not (relative_residual < limit and max_solution_error < limit):
                raise AssertionError((relative_residual, max_solution_error))
            record("solution_verified")
        # ExitStack destroys solver, vectors, matrix, resources, config in order.
        record("amgx_objects_destroyed")
    finally:
        pyamgx.finalize()
    record("amgx_finalized")

    nnz = csr.nnz
    del csr, rhs, solution
    gc.collect()
    record("cupy_arrays_deleted")
    pool.free_all_blocks()
    record("cupy_pool_released")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n": n, "nnz": nnz, "mode": args.mode,
        "amgx_pool": "disabled (4 KiB reserve)" if args.no_amgx_pool else "default",
        "config": config, "buffers": payload,
        "shared_payload_bytes": sum(item["bytes"] for item in payload.values()),
        "native_pointers": pointer_checks, "all_five_pointers_match": True,
        "relative_residual": relative_residual, "max_solution_error": max_solution_error,
        "measurements": measurements,
        "measurement_notes": [
            "Synchronized stage snapshots, not transient peak measurements.",
            "CUDA device memory includes other processes, context, libraries and allocator pools.",
            "CuPy pool counters exclude allocations made directly by AMGX.",
            "A zero device delta can mean reuse of reserved memory; pointer checks establish aliasing.",
            "Freeing arrays/pools does not destroy the CUDA context or unload all GPU libraries.",
        ],
        "environment": {
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy_version": cp.__version__, "pyamgx_extension": Path(pyamgx.__file__).name,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "amgx_libraries": sorted({Path(line.split()[-1]).name for line in
                Path("/proc/self/maps").read_text().splitlines() if "libamgx" in line}),
        },
    }
    print(f"\n{n} rows, {nnz} nonzeros, {args.mode}; AMGX pool: {report['amgx_pool']}")
    print(f"Shared payload: {report['shared_payload_bytes']:,} bytes; all 5 pointers match.")
    print(f"Relative residual: {relative_residual:.3e}; max |x - 1|: {max_solution_error:.3e}")
    print("\nGPU memory in MiB; device usage is device-wide; delta is from the previous row.")
    print(f"{'Stage':<25} {'Device used':>12} {'Delta':>10} {'CuPy live':>12} {'CuPy reserved':>14}")
    for row in measurements:
        mib = 1024**2
        print(f"{row['stage']:<25} {row['device_used_bytes']/mib:12.3f} "
              f"{row['device_delta_previous_bytes']/mib:+10.3f} "
              f"{row['cupy_live_bytes']/mib:12.3f} {row['cupy_reserved_bytes']/mib:14.3f}")
    print("Stage snapshots exclude transient peaks; reserved pools can hide new allocations.")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nReport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
