# Sharing CuPyX CSR storage with AMGX

Status: all three native CSR regressions, 136 Python tests, and 43 PTDS attachment
tests pass. FP32/FP64 transfer profiles confirm sharing of all three CSR buffers,
and the small FP64 memory check reports zero errors. See
[validation record](borrowed_csr_validation.md) for the qualified scope.

For direct `solver.setup(csr)` / `solver.solve(rhs, out=solution)` calls with
automatic handle reuse and explicit cleanup, see [ReusableSolver](reusable_solver.md).

```python
A = pyamgx.Matrix().create(resources, mode="dDDI").attach_CSR(csr)
b = pyamgx.Vector().create(resources, mode="dDDI").attach(rhs)
x = pyamgx.Vector().create(resources, mode="dDDI").attach(solution)
solver = pyamgx.Solver().create(resources, config, mode="dDDI")
solver.setup(A)
solver.solve(b, x)
assert A.attached_ptrs == (csr.indptr.data.ptr, csr.indices.data.ptr, csr.data.data.ptr)
# The solution is already in `solution`; no download is necessary.
solver.destroy()
assert A.detach() is csr
```

`attach_CSR` borrows all three buffers. `upload_CSR` continues to copy them.
The matrix retains the CSR object and each array independently; replacing a
buffer on the CSR object cannot free memory still used by AMGX. Before setup or
solve, the wrapper checks buffer identity, pointer, extent, and shape, then waits
for each exported producer stream. CUDA Array Interface v2 requires an explicit
integer `stream=`; v3 provides stream metadata. Call from the producer's stream
context, or explicitly supply a stream ordered after all writes. Methods are
synchronous; concurrent access to shared buffers is prohibited.

Use square canonical scalar CSR with native int32 indices and values matching
the matrix mode: float64/dDDI or float32/dFFI. Every row must contain its explicit
diagonal entry. Empty matrices, missing diagonals, duplicates, unsorted columns,
wrong-device/nondevice pointers, incompatible dtypes, and noncontiguous buffers
are rejected. There is no implicit canonicalization or fallback copy. Prepare
the desired CSR representation before attachment.

Keep structure immutable while attached. To update coefficients, write
`csr.data` in place, then rerun `solver.setup(A)` in the producer stream context.
For value updates use `structure_reuse_levels=0`; AMGX's `-1` setting explicitly
reuses the entire hierarchy and skips setup, leaving preconditioner data stale.
Destroy associated solvers before detach/destroy. The wrapper conservatively
retains all matrices passed to setup until solver destruction, including failed
setup attempts. A failed setup disables solve until a later successful setup.
AMGX operations that scale/reorder/insert into fine CSR storage are rejected.
dDFI retains the previously documented mixed-precision SpMV limitation.

AMGX may allocate diagonal metadata, validation scratch, solver workspaces, and
coarse levels. GPU validation returns a scalar status to the host; Krylov methods
may also transfer scalar reductions. The zero-copy claim concerns the three
supplied fine CSR buffers, not all internal solver data.

For a standalone solve passing a CuPyX CSR matrix and CuPy vectors directly,
see [the memory example and recorded measurements](cupy_csr_memory.md). It checks
all five shared pointers and reports GPU memory at attachment, setup, solve,
and cleanup, with an optional diagnostic that disables AMGX pooling.

## Validation commands

Build AMGX with the attachment APIs, then follow the portable
[Python build and runtime setup](gpu_setup.md). Commands below run from the
PyAMGX repository root; no application-specific launcher is needed.

Run `tests/test_matrix_attach.py` together with the existing vector, matrix,
solver, and protocol tests. Repeat attachment tests with
`CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1`. These are small matrices, not simulations.

Profile `examples/borrowed_csr.py --copy-baseline --report=results/borrowed-csr.json`
with Nsight Systems CUDA/NVTX tracing, for both `--mode=dDDI` and `--mode=dFFI`.
The fixture checks native/CuPy pointer identity and independent residuals. Inspect
`csr.attach`, `csr.setup`, `csr.solve`, `csr.detach`, and `csr.copy_upload` ranges.
The copy control should expose three full CSR transfers. Attachment may copy the
validation scalar but must not stage any CSR array. Record results before marking
this milestone complete.
