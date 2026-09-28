# One million DOFs with AMG preconditioning

The [benchmark](../examples/benchmark_amg_million.py) ran successfully on
2026-09-28 using an RTX PRO 5000 Blackwell, CuPy 14.2.0, and the existing
CUDA 13 AMGX library. [Raw measurements and native hierarchy diagnostics](reusable_solver_amg_million.json)
are recorded alongside this document. No AMGX rebuild was performed.

The matrix is a 100 × 100 × 100 unscaled seven-point Dirichlet Laplacian plus
`0.1 I`: **1,000,000 rows, 6,940,000 nonzeros**, FP64. Five seeded random exact
solutions generate distinct RHS through `b = A @ exact`. Each solve starts
from zero, while the matrix and all five borrowed buffers stay fixed.

The solver is FGMRES, restart 30, tolerance `1e-8`, preconditioned by one
classical AMG V-cycle. AMG uses PMIS coarsening, D2 interpolation, two pre/post
JACOBI_L1 sweeps, at most 15 levels, and a dense LU coarse solver. Explicit
`structure_reuse_levels=0` requests a full hierarchy reconstruction whenever
`setup()` is called. It does not cause reconstruction during `solve()`.

## Timings

| Workflow | Median milliseconds per RHS |
| --- | ---: |
| Explicit Matrix/Vector/Solver API, setup once | 34.21 |
| ReusableSolver, setup once | 30.34 |
| ReusableSolver, rebuild AMG before each RHS | 100.66 |

Reusing AMG was **3.32× faster** than rebuilding before each RHS in this run,
reducing the measured per-RHS latency by about 70%. The median reconstruction
call alone was **73.21 ms**. The explicit API also supports setup reuse; the
30 versus 34 ms comparison does not establish an intrinsic wrapper speedup.

Each workflow has ten timed solves: five distinct RHS in each of two phases.
The workflows run in forward then reverse order. Construction, initial setup,
and one warmup solve per phase are outside these timings. Initial setup took
73.6–105.9 ms across phases; it is still paid once when reusing AMG. RHS copying,
output zeroing, and validation are excluded. All timings synchronize GPU work.
Reuse phase medians ranged from 27.7 to 35.2 ms; reconstruction phase medians
were 100.5 and 102.9 ms, so the reuse benefit is clear despite timing variation.

All 30 timed solves returned success in **9 iterations**. The largest independent
relative residual was `1.296e-9`, and the largest relative solution error was
`3.213e-9`. Full CSR comparisons confirmed no row-offset, column-index, or
coefficient mutations during the timed phases. All five native pointers matched
the CuPy storage.

## Reconstruction after changing coefficients

A separate diagnostic enables native AMG hierarchy reporting. The shift is
changed in place from **0.1 to 1.0**, and `solver.setup(csr)` is called again on
the same reusable solver. The GPU buffer addresses remain unchanged.

| Operation | Native hierarchy reports | Levels | Iterations | Relative residual |
| --- | ---: | ---: | ---: | ---: |
| Initial setup, shift 0.1 | 1 | 5 | — | — |
| Solve without setup | 0 | — | 9 | 1.246e-9 |
| Setup after shift 1.0 | 1 | 6 | — | — |
| Solve after reconstruction | 0 | — | 8 | 9.652e-10 |
| Fresh solver setup, shift 1.0 | 1 | 6 | — | — |
| Fresh solver solve | 0 | — | 8 | 9.652e-10 |

The hierarchy changed from row counts
`[1000000, 306065, 35541, 2896, 286]` to
`[1000000, 306065, 35926, 4168, 1433, 439]`.
Reconstruction took **76.79 ms**, versus **77.11 ms** for fresh setup in this
separate diagnostic. The reconstructed and fresh hierarchy reports matched,
and the relative difference between their solutions was **0.0** in this run.
This verifies that setup actually reconstructed the hierarchy after the value
change. The script asserts these diagnostics, convergence, and pointer equality.
Native source also routes `structure_reuse_levels=0` resetup through deletion
and recreation of the existing finest AMG level (`src/amg.cu`, around line 914).

Use this pattern:

```python
with pyamgx.ReusableSolver(config) as solver:
    solver.setup(csr)               # Construct AMG once.
    solver.solve(rhs1, out=x)
    solver.solve(rhs2, out=x)       # Reuse AMG; x is the initial guess.
    # After updating csr.data in place:
    solver.setup(csr)               # Reconstruct AMG for the new coefficients.
    solver.solve(updated_rhs, out=x)
```

Application code still owns `pyamgx.initialize()` and `pyamgx.finalize()`.
The wrapper does not detect coefficient changes automatically. Reconstructing
AMG for every unchanged-matrix RHS would remove the measured reuse benefit.

## Memory

The supplied CSR, RHS, and output occupy **98.50 MiB**. CuPy live memory remained
**258.03 MiB**, and its reserved pool remained **453.09 MiB** throughout all timed
phases. Those totals include reference CSR copies and exact-solution/RHS banks
used for verification, not just solver inputs.

Device-wide memory grew by about **1,074 MiB** at setup after the first cold
phase. Further increases were observed during the solve-and-validation loops:
about 54–130 MiB in the reuse phases and 214 MiB in rebuild phases. Destruction
released about 1,174 MiB for reuse phases and 1,288 MiB for rebuild phases.
These snapshots include native allocation pools, runtime activity, and other
GPU users; they are neither isolated AMG allocation counts nor transient peak
measurements. Constant CuPy counters alone do not imply constant native memory.

## Reproduce

Follow [build and runtime setup](gpu_setup.md), then run from the repository root:

```bash
python examples/benchmark_amg_million.py --report results/amg-million.json
```

The script imports shared measurement helpers from the adjacent
`benchmark_dilu_reuse.py`. Increase `--rhs-count` (default 5) for more samples;
this also increases reference-array memory. This synthetic shifted problem
is not a prediction for every Laplacian scaling or production system.
