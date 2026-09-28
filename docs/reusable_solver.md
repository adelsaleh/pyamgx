# Reusable zero-copy CuPy solver

`ReusableSolver` owns the AMGX handles and accepts a CuPyX CSR matrix and CuPy
RHS/output arrays directly. A required `out=` makes output allocation explicit.
The wrapper does not import CuPy or allocate temporary GPU arrays.

```python
pyamgx.initialize()
try:
    with pyamgx.ReusableSolver(config, mode="dDDI") as solver:
        solver.setup(csr)
        solver.solve(rhs, out=solution)
        # Change rhs in place, then reuse the same setup and all handles.
        rhs *= 2
        solver.solve(rhs, out=solution)
finally:
    pyamgx.finalize()
```

`solve` returns the original `out` object. Its current contents are the initial
guess, so repeated calls naturally allow warm starts. Use `solution.fill(0)`
before a call if you want a fresh solve. The wrapper deliberately does not expose
the native zero-initial-guess flag: this AMGX FGMRES implementation still reads
the existing output when that flag is set.

Explicit cleanup is equally supported:

```python
solver = pyamgx.ReusableSolver(config, mode="dDDI")
try:
    solver.setup(csr)
    solver.solve(rhs, out=solution)
finally:
    solver.destroy()
```

`destroy()` is idempotent. It destroys the native solver before the vectors and
matrix, releases retained producer references, and releases managed resources
when their last solver is destroyed. Caller-owned arrays remain valid.
`setup`/`solve` after destruction raise `RuntimeError`. Application code owns
global `initialize()`/`finalize()`; destroy all solvers before finalizing.

The repeated-solve path compares the RHS/output objects by identity. Unchanged
objects remain attached; changed objects are detached and rebound with no
payload copy. Native handles and matrix setup are reused. The existing
descriptor, dimension, overlap and producer-stream checks remain active;
identity alone is not treated as proof that storage is unchanged.

After updating coefficients in place, call `solver.setup(csr)` again. This reuses
the matrix and solver handles and refreshes setup. Replacing the CSR object or
its buffers causes setup to destroy the previous native solver, rebind the
matrix, and create fresh solver state, releasing previous matrix owners.
Structure must remain immutable between setup calls; in-place structural edits
while borrowed are unsupported. Buffer replacement should be followed by setup
before another solve. A failed setup disables solve until a successful setup.
The caller must request setup after coefficient writes: GPU value changes are
not automatically detected.

Configuration is copied at construction and requires a dictionary with a
`solver` dictionary, `config_version=2`, and `exception_handling=0` (the latter
two default to those values). Changing the caller's dictionary does not alter
an existing solver. `structure_reuse_levels=-1` is rejected because it skips
coefficient refresh. Use `0` for coefficient updates. Modes are `dDDI` (float64)
and `dFFI` (float32); the [borrowed CSR contract](borrowed_csr.md) applies.

Live reusable solvers share one managed AMGX resource handle because destroying
separate native resources releases global device pools/BLAS state. Their
top-level configuration must match; solver-specific choices may differ inside
the `solver` dictionary. Destroying one reusable solver leaves the others
usable. Do not independently destroy low-level AMGX resources while reusable
solvers are alive. This API is for serialized, single-GPU use; concurrent or
reentrant managed operations are rejected while a solver is busy.

Call setup/solve from the producer stream context. An optional integer `stream=`
on either method overrides producer metadata for all buffers in that call;
CUDA array-interface v2 producers require it. Keep the stream alive and order
all writes before the call. The methods are synchronous and do not permit
concurrent buffer access. `solver.attached_ptrs` exposes native CSR/RHS/output
pointers for diagnostics; `status` and `iterations_number` expose the last
completed solve's result. Checking these properties is outside the hot path.

The runnable [two-row example](../examples/reusable_solver.py) demonstrates
repeated solves and cleanup. Follow [build and runtime setup](gpu_setup.md),
then run from the PyAMGX repository root:

```bash
python examples/reusable_solver.py
```

The [benchmark](../examples/benchmark_reusable_solver.py) compares warmed solves
against the explicit `Matrix`/`Vector`/`Solver` API, with identical arrays,
configuration and output zeroing on every call. Initialization and setup are
outside the solve timer. It runs explicit/reusable/reusable/explicit phases,
checks native pointer equality and independent residuals, and records CuPy
memory counters around each timed phase. Run it with the same environment:

```bash
python examples/benchmark_reusable_solver.py \
  --report results/pyamgx-reusable-benchmark.json
```

Validated on 2026-09-28 with an RTX PRO 5000 Blackwell and CuPy 14.2.0:

| Rows | Explicit API | Reusable API | AMGX iterations per solve |
| --- | ---: | ---: | ---: |
| 2 | 156.26 microseconds | 142.27 microseconds | 1 |
| 4,096 | 2.660 milliseconds | 2.629 milliseconds | 16 |

These are medians of ten batch means per API and size, 100 solves per batch,
after 20 warmup solves per phase. Timings include identical CuPy output zeroing
and synchronous GPU execution. The wrapper shows no material added cost in
this comparison; the small apparent speedups should not be interpreted as a
guaranteed improvement. Matrix/RHS/solution pointers match in all phases, and
CuPy live/reserved memory counters remain constant during the timed solves.
One two-row reusable phase recorded an 8 MiB device-wide memory increase; the
other phases recorded zero. That counter includes other processes and native
allocators, and cannot isolate allocation activity within AMGX's pool.
See the [raw benchmark record](reusable_solver_benchmark.json).

The same benchmark was also run with 20,000 rows and 59,998 nonzeros, using an
FP64 `tridiag(-1, 4, -1)` matrix, FGMRES without a preconditioner, and an all-ones
exact solution. Both APIs took 15 iterations from a zero initial guess and
achieved a relative residual of `7.068e-11` and maximum solution error of
`3.649e-9`.

| 20,000-row run | Explicit API | Reusable API | Relative difference |
| --- | ---: | ---: | ---: |
| Initial | 3.379 ms | 3.903 ms | +15.51% |
| Repeat | 3.532 ms | 3.537 ms | +0.14% |

The initial run had substantial timing drift: the explicit phases had medians
of 4.125 ms and 2.753 ms. One repeat with the same settings gave much more
consistent phases and only a 5-microsecond median difference between APIs.
These measurements do not establish a repeatable wrapper slowdown. Both runs
are retained: [initial record](reusable_solver_benchmark_20k.json) and
[repeat record](reusable_solver_benchmark_20k_repeat.json). Each run includes
1,000 timed solves per API, with initialization/setup excluded and the output
cleared before each solve. This is a simple, well-conditioned tridiagonal test;
row count alone does not predict performance on other sparse systems.

All five native pointers matched the CuPy buffers in every phase. CuPy live
memory stayed at 1,121,280 bytes (1.069 MiB), and its reserved pool stayed at
4,078,080 bytes (3.889 MiB) throughout all timed phases. The supplied CSR, RHS,
and output buffers contain 1,119,980 bytes. These counters exclude native AMGX
workspace and initial resource reservations. Device-wide memory changes were
`[0, 0, 327680, 0]` bytes in the initial run and
`[2883584, 0, -2883584, 0]` bytes in the repeat, in phase order. Device-wide
counters include other processes and do not measure transient allocation peaks.

To reproduce the 20,000-row run with the environment above:

```bash
python examples/benchmark_reusable_solver.py \
  --sizes 20000 --report results/pyamgx-reusable-benchmark-20k.json
```

Validation: 155 Python regressions passed, including 19 new reusable-solver
cases. All 19 also passed with `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1`.
The 19-case suite passed Compute Sanitizer memcheck with zero errors using:

```bash
compute-sanitizer \
  --tool memcheck --show-backtrace no --error-exitcode 99 \
  python -m pytest -q \
  tests/test_reusable_solver.py
```

With default sanitizer backtrace collection, three weak-reference lifetime
assertions failed despite zero GPU memory errors. Disabling backtrace collection
allowed those same assertions to pass; all memory-access checks remained active.
Normal and PTDS runs pass without this instrumentation setting. The tests cover
FP32/FP64, streams, explicit v2 producer streams, changed buffers/dimensions,
coefficient refresh, native setup and constructor failures, caller ownership,
context exceptions, double destroy, garbage collection, reentrancy rejection,
and destruction of one solver while another remains live.

The recorded checks used an isolated Python/Cython extension and an existing
`libamgxsh.so`. Raw test logs are not bundled; benchmark JSON records are linked
above. Published paths are generic, and measured values are unchanged.

For larger preconditioned cases, see the [DILU benchmark with 120k–160k rows](reusable_solver_dilu.md), including changing RHS, setup-refresh comparisons, and memory measurements.

The [million-DOF AMG benchmark](reusable_solver_amg_million.md) verifies setup reuse and hierarchy reconstruction after an in-place coefficient change.
