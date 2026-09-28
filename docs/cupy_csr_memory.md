# CuPyX/CuPy solve with GPU memory measurements

[Runnable example](../examples/cupy_csr_memory.py), run on 2026-09-28 with an
RTX PRO 5000 Blackwell, CuPy 14.2.0 and the existing CUDA 13 AMGX build.
The example creates a 4,096-row tridiagonal CSR matrix with diagonal 4 and
off-diagonals -1, a CuPy RHS, and a zeroed CuPy solution array. The exact solution
is all ones. It uses FGMRES with `NOSOLVER` as its preconditioner.

The objects go directly to PyAMGX:

```python
A.attach_CSR(csr)     # cupyx.scipy.sparse.csr_matrix
b.attach(rhs)        # cupy.ndarray
x.attach(solution)   # cupy.ndarray
solver.setup(A)
solver.solve(b, x)    # writes into the original `solution` array
```

The script checks all five native pointers against the original CuPy pointers
after attachment, setup, and solve. Both FP64 runs passed, sharing 229,356 bytes
of CSR/vector data. The independent relative residual was `4.1825648e-11` and
maximum error against the known solution was `9.7657771e-10`.

Follow [build and runtime setup](gpu_setup.md), then run in a fresh process
from the PyAMGX repository root:

```bash
python examples/cupy_csr_memory.py --report results/csr-memory.json
```

The script prints the memory table and saves exact byte counts, pointers,
configuration, numerical checks, and binary filenames in JSON.
`--n 3` gives a three-row example; `--mode dFFI` selects FP32.

To observe allocations with AMGX pooling disabled, repeat the same command with
`--no-amgx-pool --report results/pyamgx-csr-memory-no-pool.json`. This build still
requires a nonzero initial pool reservation; the diagnostic uses 4 KiB. AMGX's
text configuration parser is used because its JSON parser rejects the `size_t`
pool-size setting.

Recorded FP64 measurements, in MiB (1 MiB = 1,048,576 bytes):

| Operation | Device memory change, default pool | Device memory change, pooling disabled |
| --- | ---: | ---: |
| Create CuPy inputs | +4 | +4 |
| Initialize AMGX | 0 | 0 |
| Create AMGX resources | +264 | +8 |
| Create empty matrix/vector/solver handles | +52 | +52 |
| Attach CSR | 0 | 0 |
| Attach RHS and solution | 0 | 0 |
| Solver setup | 0 | 0 |
| Solve | 0 | +2 |
| Verify solution | 0 | 0 |
| Destroy AMGX objects/resources | -264 | -10 |
| Finalize AMGX | 0 | 0 |
| Delete CuPy arrays | 0 | 0 |
| Release CuPy cached blocks | -2 | -2 |

These are changes between consecutive synchronized snapshots. The largest
observed increase above the initial CUDA-context snapshot was 320 MiB with the
default pool and 66 MiB with pooling disabled. The default AMGX pool reserves
256 MiB before attachment. CuPy's live allocation count remained 229,888 bytes
(0.219 MiB) from input creation through AMGX destruction; its reserved pool was
841,216 bytes (0.802 MiB). CuPy live usage became zero after deleting the arrays,
and reserved usage became zero after releasing cached blocks.

Device-wide memory includes other applications, allocator reservations, CUDA
context and library costs; baseline totals therefore differ between runs. The
CuPy pool counters exclude native AMGX allocations. Zero change at attachment
or setup can reflect reuse of reserved memory or CUDA allocation granularity.
Pointer identity verifies that the supplied buffers are shared. These stage
snapshots do not capture transient peaks, and they do not quantify allocations
within AMGX's pool. CUDA context and library memory can remain resident until
the process exits.

The raw measurements are saved in [default-pool report](cupy_csr_memory_default.json)
and [pooling-disabled report](cupy_csr_memory_no_pool.json). The example also
passed with `--mode dFFI`. See [the attachment contract](borrowed_csr.md) for
supported CSR formats, configurations, and buffer lifetimes.
