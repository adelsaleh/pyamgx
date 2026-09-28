"""Fixed GPU CSR, changing RHS: DILU setup reuse versus setup per RHS.

Runs diagnostics only, using the existing AMGX library. No native build.
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

CONFIG = {
    'config_version': 2, 'exception_handling': 0, 'determinism_flag': 1,
    'solver': {
        'solver': 'FGMRES', 'max_iters': 500, 'gmres_n_restart': 30,
        'monitor_residual': 1, 'convergence': 'RELATIVE_INI_CORE',
        'tolerance': 1e-8,
        'preconditioner': {
            'solver': 'MULTICOLOR_DILU', 'max_iters': 1,
            'reorder_cols_by_color': 0, 'insert_diag_while_reordering': 0,
            'coloring_level': 1, 'matrix_coloring_scheme': 'MIN_MAX',
            'max_uncolored_percentage': 0.0,
        },
    },
}


def snapshot():
    cp.cuda.runtime.deviceSynchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return dict(cupy_live_bytes=pool.used_bytes(), cupy_reserved_bytes=pool.total_bytes(),
                device_used_bytes=total - free)


def matrix(case):
    if case == 'tridiagonal_120k':
        n = 120000
        return sparse.diags([cp.full(n-1, -1.), cp.full(n, 4.), cp.full(n-1, -1.)],
                            [-1, 0, 1], format='csr')
    side, dim = (400, 2) if case == 'shifted_2d_160k' else (50, 3)
    t = sparse.diags([cp.full(side-1, -1.), cp.full(side, 2.), cp.full(side-1, -1.)],
                     [-1, 0, 1], format='csr')
    a = t
    for _ in range(dim - 1):
        a = sparse.kronsum(a, t, format='csr')
    return (a + 0.1 * sparse.eye(a.shape[0], format='csr')).tocsr()


def phase(kind, a, rhs_bank, exact_bank, rhs, out, originals):
    memory = {'before_resources': snapshot()}
    with ExitStack() as cleanup:
        def own(obj):
            cleanup.callback(obj.destroy)
            return obj
        start = perf_counter()
        if kind == 'explicit_reuse':
            cfg = own(pyamgx.Config().create_from_dict(CONFIG))
            resources = own(pyamgx.Resources().create_simple(cfg))
            A = own(pyamgx.Matrix().create(resources, 'dDDI')).attach_CSR(a)
            b = own(pyamgx.Vector().create(resources, 'dDDI')).attach(rhs)
            x = own(pyamgx.Vector().create(resources, 'dDDI')).attach(out)
            solver = own(pyamgx.Solver().create(resources, cfg, 'dDDI'))
            setup = lambda: solver.setup(A)
            solve = lambda: solver.solve(b, x)
            pointers = lambda: (*A.attached_ptrs, b.attached_ptr, x.attached_ptr)
        else:
            solver = cleanup.enter_context(pyamgx.ReusableSolver(CONFIG, mode='dDDI'))
            setup = lambda: solver.setup(a)
            solve = lambda: solver.solve(rhs, out=out)
            def pointers():
                p = solver.attached_ptrs
                return (*p['csr'], p['rhs'], p['solution'])
        cp.cuda.runtime.deviceSynchronize()
        construct_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        setup()
        cp.cuda.runtime.deviceSynchronize()
        first_setup_ms = (perf_counter() - start) * 1000
        memory['after_setup'] = snapshot()
        cp.copyto(rhs, rhs_bank[0])
        out.fill(0)
        solve()  # Warm kernels and first vector attachments, outside timings.
        memory['after_warmup'] = snapshot()
        expected = tuple(v.data.ptr for v in (a.indptr, a.indices, a.data, rhs, out))
        assert pointers() == expected
        samples = []
        for i, source in enumerate(rhs_bank):
            cp.copyto(rhs, source)
            out.fill(0)
            cp.cuda.runtime.deviceSynchronize()
            start = perf_counter()
            if kind == 'setup_each_rhs':
                setup()
            cp.cuda.runtime.deviceSynchronize()
            setup_end = perf_counter()
            solve()
            cp.cuda.runtime.deviceSynchronize()
            end = perf_counter()
            residual = float(cp.linalg.norm(a @ out - rhs) / cp.linalg.norm(rhs))
            error = float(cp.linalg.norm(out - exact_bank[i]) / cp.linalg.norm(exact_bank[i]))
            assert residual < 2e-8 and error < 1e-5, (kind, residual, error)
            assert pointers() == expected
            samples.append(dict(rhs_index=i, setup_ms=(setup_end-start)*1000,
                                solve_ms=(end-setup_end)*1000, total_ms=(end-start)*1000,
                                iterations=solver.iterations_number, status=str(solver.status),
                                relative_residual=residual, relative_solution_error=error))
        memory['after_solves'] = snapshot()
        assert all(bool(cp.array_equal(v, original)) for v, original in
                   zip((a.indptr, a.indices, a.data), originals)), 'CSR contents changed'
        memory['after_validation'] = snapshot()
    memory['after_destroy'] = snapshot()
    return dict(kind=kind, construct_ms=construct_ms, first_setup_ms=first_setup_ms,
                samples=samples, all_five_pointers_match=True, csr_unchanged=True, memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=['tridiagonal_120k', 'shifted_2d_160k',
                        'shifted_3d_125k'], default=['tridiagonal_120k', 'shifted_2d_160k', 'shifted_3d_125k'])
    parser.add_argument('--rhs-count', type=int, default=5)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.rhs_count < 2:
        parser.error('at least two changing right-hand sides are required')
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
                  gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  cupy_version=cp.__version__, pyamgx_extension=Path(pyamgx.__file__).name, results=[],
                  notes=['FP64 fixed CSR; different seeded random exact solutions give b=A*x.',
                         'Same RHS sequence per phase; zero output before each solve; no warm starts.',
                         'RHS copy, output zeroing and residual checks excluded from timings.',
                         'Setup-each refreshes the same handles, not full resource recreation.',
                         'Forward then reverse phase order; one warmup solve per phase.',
                         'Memory snapshots include validation arrays; no transient peak measurement.',
                         'Device-wide memory includes other processes and native AMGX reservations.'])
    pyamgx.initialize()
    try:
        for case in args.cases:
            a = matrix(case)
            a.sum_duplicates()
            a.sort_indices()
            n = a.shape[0]
            rng = cp.random.RandomState(2026)
            exacts = [rng.standard_normal(n) for _ in range(args.rhs_count)]
            sources = [a @ exact for exact in exacts]
            rhs, out = cp.empty(n), cp.zeros(n)
            originals = [v.copy() for v in (a.indptr, a.indices, a.data)]
            phases = []
            for kind in ['explicit_reuse', 'reusable_reuse', 'setup_each_rhs',
                         'setup_each_rhs', 'reusable_reuse', 'explicit_reuse']:
                p = phase(kind, a, sources, exacts, rhs, out, originals)
                phases.append(p)
                print(case, kind, 'median ms:', round(median(s['total_ms'] for s in p['samples']), 3), flush=True)
            times = {kind: median(s['total_ms'] for p in phases if p['kind'] == kind for s in p['samples'])
                     for kind in ('explicit_reuse', 'reusable_reuse', 'setup_each_rhs')}
            report['results'].append(dict(case=case, rows=n, nnz=a.nnz,
                borrowed_payload_bytes=sum(v.nbytes for v in (a.indptr, a.indices, a.data, rhs, out)),
                median_ms_per_rhs=times, phases=phases))
            args.report.write_text(json.dumps(report, indent=2) + '\n')
            del a, sources, exacts, rhs, out, originals
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        pyamgx.finalize()
    print('Report:', args.report.resolve(), flush=True)


if __name__ == '__main__':
    main()
