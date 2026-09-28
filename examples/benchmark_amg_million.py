"""Million-DOF shifted Laplacian: AMG reuse, rebuild cost, coefficient refresh.

Uses the existing AMGX library; no builds or time integration.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import median
from time import perf_counter

import cupy as cp
from cupyx.scipy import sparse
import pyamgx
import benchmark_dilu_reuse as common

CONFIG = {
    'config_version': 2, 'exception_handling': 0, 'determinism_flag': 1,
    'solver': {
        'scope': 'main', 'solver': 'FGMRES', 'max_iters': 200,
        'gmres_n_restart': 30, 'monitor_residual': 1,
        'convergence': 'RELATIVE_INI_CORE', 'tolerance': 1e-8,
        'preconditioner': {
            'scope': 'amg', 'solver': 'AMG', 'algorithm': 'CLASSICAL',
            'selector': 'PMIS', 'interpolator': 'D2',
            'max_iters': 1, 'max_levels': 15, 'min_coarse_rows': 32,
            'cycle': 'V', 'presweeps': 2, 'postsweeps': 2,
            'smoother': {'scope': 'smooth', 'solver': 'JACOBI_L1'},
            'coarse_solver': 'DENSE_LU_SOLVER',
            'structure_reuse_levels': 0,
        },
    },
}


def coefficient_check(a, exact, rhs, out, side):
    config = deepcopy(CONFIG)
    config['solver']['preconditioner']['print_grid_stats'] = 1
    messages = []
    pyamgx.register_print_callback(messages.append)

    def operation(name, fn):
        messages.clear()
        cp.cuda.runtime.deviceSynchronize()
        start = perf_counter()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        elapsed = (perf_counter() - start) * 1000
        text = ''.join(messages)
        return dict(operation=name, elapsed_ms=elapsed,
                    hierarchy_reports=text.count('Number of Levels:'), native_output=text)

    def accuracy(solver):
        residual = float(cp.linalg.norm(a @ out - rhs) / cp.linalg.norm(rhs))
        error = float(cp.linalg.norm(out - exact) / cp.linalg.norm(exact))
        assert solver.status == 'success' and residual < 2e-8 and error < 1e-5
        return dict(relative_residual=residual, relative_solution_error=error,
                    status=solver.status, iterations=solver.iterations_number)

    events = []
    with pyamgx.ReusableSolver(config) as solver:
        events.append(operation('initial_setup_shift_0.1', lambda: solver.setup(a)))
        cp.copyto(rhs, a @ exact)
        out.fill(0)
        events.append(operation('solve_without_setup', lambda: solver.solve(rhs, out=out)))
        before = accuracy(solver)
        ptrs = solver.attached_ptrs
        # Sorted lexicographic 3D stencil: diagonal follows existing negative neighbors.
        rows = cp.arange(a.shape[0], dtype=cp.int32)
        offsets = (rows % side > 0).astype(cp.int32)
        offsets += (rows // side % side > 0)
        offsets += (rows // (side * side) > 0)
        diagonal = a.indptr[:-1] + offsets
        assert bool(cp.all(a.indices[diagonal] == rows))
        a.data[diagonal] += 0.9  # Shift 0.1 -> 1.0, same structure and allocations.
        cp.copyto(rhs, a @ exact)
        out.fill(0)
        events.append(operation('rebuild_after_shift_1.0', lambda: solver.setup(a)))
        events.append(operation('solve_after_rebuild', lambda: solver.solve(rhs, out=out)))
        updated = accuracy(solver)
        saved_solution = out.copy()
        assert solver.attached_ptrs == ptrs
        with pyamgx.ReusableSolver(config) as fresh:
            events.append(operation('fresh_setup_shift_1.0', lambda: fresh.setup(a)))
            out.fill(0)
            events.append(operation('fresh_solve', lambda: fresh.solve(rhs, out=out)))
            fresh_accuracy = accuracy(fresh)
            difference = float(cp.linalg.norm(out - saved_solution) / cp.linalg.norm(out))
            assert difference < 1e-7
        setups = [events[i] for i in (0, 2, 4)]
        solves = [events[i] for i in (1, 3, 5)]
        assert all(e['hierarchy_reports'] == 1 for e in setups)
        assert all(e['hierarchy_reports'] == 0 for e in solves)
        levels = [int(re.search(r'Number of Levels:\s+(\d+)', e['native_output'])[1])
                  for e in setups]
        assert levels[0] != levels[1], 'shift change did not change hierarchy depth'
        assert setups[1]['native_output'] == setups[2]['native_output'], 'fresh hierarchy differs'
    pyamgx.register_print_callback(lambda text: None)
    return dict(events=events, initial=before, updated=updated, fresh=fresh_accuracy,
                relative_difference_updated_vs_fresh=difference, all_five_pointers_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rhs-count', type=int, default=5)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.rhs_count < 2:
        parser.error('use at least two right-hand sides')
    side = 100
    t = sparse.diags([cp.full(side-1, -1.), cp.full(side, 2.), cp.full(side-1, -1.)],
                     [-1, 0, 1], format='csr')
    a = sparse.kronsum(sparse.kronsum(t, t, format='csr'), t, format='csr')
    a = (a + 0.1 * sparse.eye(a.shape[0], format='csr')).tocsr()
    a.sum_duplicates()
    a.sort_indices()
    n = a.shape[0]
    rng = cp.random.RandomState(2026)
    exacts = [rng.standard_normal(n) for _ in range(args.rhs_count)]
    sources = [a @ exact for exact in exacts]
    rhs, out = cp.empty(n), cp.zeros(n)
    originals = [v.copy() for v in (a.indptr, a.indices, a.data)]
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy_version=cp.__version__,
        pyamgx_extension=Path(pyamgx.__file__).name, config=CONFIG, rows=n, nnz=a.nnz,
        rhs_count=args.rhs_count, side=side, dimension=3, shift=0.1,
        borrowed_payload_bytes=sum(v.nbytes for v in (a.indptr,a.indices,a.data,rhs,out)),
        phases=[], notes=[
            'Unscaled seven-point Dirichlet Laplacian + 0.1 I; FP64.',
            'Different seeded random exact solutions, b=A*x; fixed CSR in timed phases.',
            'Zero initial guess each solve; RHS copy, zeroing and validation excluded from timer.',
            'Forward and reverse phase order; one warmup solve per phase.',
            'structure_reuse_levels=0 requests full hierarchy rebuild on setup.',
            'Initial construction/setup outside table; setup_each_rhs includes setup before every RHS.',
            'Memory snapshots include reference arrays and native reservations, not transient peaks.',
            'Hierarchy print diagnostics enabled only in separate coefficient-refresh check.'])
    common.CONFIG = CONFIG
    pyamgx.initialize()
    try:
        report['amgx_libraries'] = sorted({Path(line.split()[-1]).name for line in
            Path('/proc/self/maps').read_text().splitlines() if 'libamgx' in line})
        for kind in ['explicit_reuse', 'reusable_reuse', 'setup_each_rhs',
                     'setup_each_rhs', 'reusable_reuse', 'explicit_reuse']:
            p = common.phase(kind, a, sources, exacts, rhs, out, originals)
            report['phases'].append(p)
            assert all(s['status'] == 'success' for s in p['samples'])
            print(kind, 'median ms:', round(median(s['total_ms'] for s in p['samples']), 3), flush=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        report['median_ms_per_rhs'] = {kind: median(s['total_ms'] for p in report['phases']
            if p['kind'] == kind for s in p['samples']) for kind in
            ('explicit_reuse', 'reusable_reuse', 'setup_each_rhs')}
        report['coefficient_refresh_check'] = coefficient_check(a, exacts[0], rhs, out, side)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print('Verified AMG reuse and coefficient refresh. Report:', args.report.resolve(), flush=True)
    finally:
        pyamgx.finalize()


if __name__ == '__main__':
    main()
