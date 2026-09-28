# DILU with more than 100,000 rows

Measured on 2026-09-28 with CUDA 13 AMGX, an isolated PyAMGX extension,
CuPy 14.2.0, and an RTX PRO 5000 Blackwell.
The [runnable benchmark](../examples/benchmark_dilu_reuse.py) and
[raw measurements](reusable_solver_dilu_benchmark.json) are included.

All cases use FP64 FGMRES with `MULTICOLOR_DILU`, one preconditioner iteration,
restart 30, maximum 500 iterations, and relative initial residual tolerance
`1e-8`. Coloring uses deterministic `MIN_MAX`, level 1. Both
`reorder_cols_by_color` and `insert_diag_while_reordering` are explicitly zero
because borrowed CSR storage must remain unchanged.

Five different seeded random exact solutions produce five right-hand sides
`b = A @ exact`. Each phase uses this same sequence, copies each RHS into the
same attached buffer, and clears the output before solving. This measures
setup reuse without warm-start benefits. All matrices remain fixed.

| Matrix | Rows | Nonzeros | Iterations | Worst relative residual |
| --- | ---: | ---: | ---: | ---: |
| `tridiag(-1, 4, -1)` | 120,000 | 359,998 | 7 | 4.62e-9 |
| 400 × 400 five-point Laplacian + 0.1 I | 160,000 | 798,400 | 29 | 9.68e-9 |
| 50 × 50 × 50 seven-point Laplacian + 0.1 I | 125,000 | 860,000 | 29–30 | 7.88e-9 |

The Laplacians use the unscaled stencil, diagonal 4 or 6 before the shift,
and Dirichlet boundaries. These are synthetic diagnostics, not application
simulations. The positive shift improves conditioning; these timings should
not be generalized to unshifted Poisson or arbitrary sparse matrices.

Median milliseconds per changing RHS:

| Rows / case | Explicit API, setup once | ReusableSolver, setup once | ReusableSolver, setup each RHS | Setup refresh alone |
| --- | ---: | ---: | ---: | ---: |
| 120,000 / tridiagonal | 1.949 | 1.896 | 2.008 | 0.416 |
| 160,000 / 2D | 20.830 | 20.624 | 20.242 | 0.613 |
| 125,000 / 3D | 21.902 | 20.954 | 21.336 | 0.647 |

Each median covers ten individual solves: five RHS in each of two phases.
The three API/workflow phases are run in forward and reverse order. Each
phase constructs fresh handles, performs initial setup, and warms up one
solve. Those initial costs are outside the table. RHS copying, output zeroing,
and independent correctness checks are also outside the timers. Every timed
solve is synchronized. Initial setup and construction timings remain in the
raw record.

`setup_each_rhs` calls setup again on the same matrix and native handles. It
measures setup refresh, not full handle recreation or CSR uploading; native
metadata such as coloring may already exist. Reuse removes the measured
0.42–0.65 ms setup call, but the total timing benefit is small and not consistent
across these short runs. In particular, the 2D setup-each total was slightly
lower than the reuse total due to variation in measured solve time. These
results do not demonstrate a reliable overall speedup for every case. The
explicit API also supports setup reuse.

All 90 timed solves returned `success`. Relative solution errors were below
`2.42e-7`. All five native pointers matched the input CuPy buffers in every
phase, and full comparisons confirmed CSR row offsets, column indices, and
coefficients were unchanged after each phase.

Memory snapshots (MiB, 2^20 bytes):

| Case | Borrowed CSR + RHS + output | CuPy live | CuPy reserved after solves |
| --- | ---: | ---: | ---: |
| Tridiagonal | 6.409 | 20.143 | 24.196 |
| 2D | 12.189 | 34.145 | 50.652 |
| 3D | 12.226 | 32.088 | 56.537 |

CuPy live usage was constant through all solves and destruction. It includes
the benchmark's reference CSR copies and five exact-solution/RHS pairs, so it
exceeds the borrowed payload. The first tridiagonal phase grew the reserved
CuPy pool by 0.916 MiB during the solve-and-validation loop; all subsequent
phases had stable reservations.

After the first cold phase, device-wide snapshots increased by 264 MiB while
AMGX resources were alive and returned to their pre-resource values after
destruction. The first cold phase reached a 320 MiB increase and retained
56 MiB after destruction. These are observed device-wide changes, including
native memory reservations and runtime effects, not isolated DILU storage or
transient peaks. CuPy counters do not include native AMGX allocations.

To reproduce without rebuilding AMGX:

Follow [build and runtime setup](gpu_setup.md), then run from the repository root:

```bash
python examples/benchmark_dilu_reuse.py --report results/dilu.json
```

Use `--cases shifted_2d_160k` to select one case, or increase `--rhs-count`
(default 5) to sample more distinct RHS. This also increases reference-array
memory usage. The original run log is not bundled; measured results are in the linked JSON record.
