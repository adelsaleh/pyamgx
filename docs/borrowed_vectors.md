# Zero-copy CUDA vector attachment

Status (2026-09-28): the rebuilt AMGX library and an isolated PyAMGX extension
pass FP32/FP64 native and GPU checks, with zero-copy boundary behavior verified
by profiling. Mixed-mode dDFI solves are outside this milestone: the existing
AMGX mixed-precision SpMV path returns NOT_IMPLEMENTED. See the
[validation record](borrowed_vectors_validation.md). Tests used an isolated
extension build.

```python
# A is an ordinary PyAMGX matrix, uploaded and set up as before.
b = pyamgx.Vector().create(resources).attach(b_cupy)
x = pyamgx.Vector().create(resources).attach(x_cupy)
solver.solve(b, x)
# x_cupy now contains the solution. No download and no vector staging copy.
assert x.attached_ptr == x_cupy.data.ptr
```

`attach` requires a one-dimensional contiguous writable CUDA-array-interface
producer and an empty vector. FP32/FP64 must match the vector mode; mixed mode
`dDFI` has FP64 vectors and FP32 matrix values. Input and output must not overlap.
The current AMGX build cannot solve in dDFI despite accepting its vector mode.
BSR integration and matrix borrowing are out of scope. Matrices still use the
existing upload API. Both vector and matrix high-level real transfers now use
their respective mode precision.

| Mode | Matrix values | RHS/solution | Contract |
| --- | --- | --- | --- |
| `dDDI` | FP64 | FP64 | Borrowed scalar solves validated |
| `dFFI` | FP32 | FP32 | Borrowed scalar solves validated |
| `dDFI` | FP32 | FP64 | Attachment works; existing SpMV returns NOT_IMPLEMENTED |

The dDFI rejection also occurs with ordinary uploaded vectors. It is in
AMGX's `src/amgx_cusparse.cu` mixed-precision `Cusparse::bsrmv` overload;
scalar CSR reaches that overload with block size 1. Copying vectors does not
fix it. Supporting mixed solves requires a separate AMGX numerical-kernel
change, rebuild, and validation. Use dDDI for float64 or dFFI for float32 to stay within the validated
scalar-solve contract. CuPyX BSR attachment is a separate unsupported case.

Mixed-mode regressions verify the expected error and that attached pointers,
RHS contents, and detach/reattach remain usable afterward. They do not claim
mixed-mode convergence or guarantee a useful solution after a failed solve.

The vector keeps a strong reference to the producer. Neither side may change
its allocation or shape while attached. `detach()` waits for AMGX completion,
returns that same producer, and leaves the native vector empty. `destroy()`
releases the borrow without freeing CuPy memory. Explicitly destroy vectors,
solvers, matrices, resources, and config before `pyamgx.finalize()`.

`upload` on an attached vector is rejected. `set_zero()` fills it in place.
`download()` remains an explicit NumPy copy API and is unnecessary for attached
solutions. `download_raw` retains its expert pointer/capacity responsibilities.

CAI v3 is the default: the wrapper reads its stream metadata at attachment and
again before each solve or zeroing operation, waiting on that stream only.
Call AMGX in the same CuPy stream context that orders the input writes:

```python
with producer_stream:
    b_cupy[...] = new_rhs
    solver.solve(b, x)
# solve is synchronous; any consumer stream can now use x_cupy.
```

This cannot discover unordered work queued on other streams. Producers must
export a stream that orders their prior uses, as required by the protocol.
For CAI v2, supply `attach(array, stream=...)`, using 1 for legacy, 2 for PTDS,
or an external handle. Keep an explicitly supplied stream alive through the
borrow. A v3 `stream=None` asserts producer work is already complete; stream
integer 0 is invalid. CUDA pointer/device validation happens in AMGX, without
requiring PyAMGX to import CuPy.

## Build and verification

Use `exception_handling=0` in the AMGX configuration so native failures return
error codes and PyAMGX can raise Python exceptions. In this AMGX version,
`exception_handling=1` makes AMGX terminate the process on native API failures.

Build AMGX with the attachment APIs, then follow the portable
[Python build and runtime setup](gpu_setup.md). From the PyAMGX repository root:

```bash
"$AMGX_BUILD_DIR/src/amgx_tests_launcher" BorrowedVectors
python -m pytest -q \
  tests/test_attach_protocol.py tests/test_vector_attach.py \
  tests/test_vector.py tests/test_matrix.py tests/test_solver.py
```

Repeat GPU tests with `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1`. CAI export-v2
is covered by explicit-stream tests; this path must not assume implicit
default-stream ordering. Record source revisions, extension/library paths,
CUDA runtime/driver, CuPy version, and GPU with results. Use profiling to
confirm no RHS/solution boundary transfers; algorithmic workspace copies and
the unchanged matrix upload must be identified separately.
