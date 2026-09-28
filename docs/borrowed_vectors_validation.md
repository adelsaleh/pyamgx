# Borrowed-vector validation — 2026-09-28

FP64 (`dDDI`) and FP32 (`dFFI`) borrowed vectors passed the tests below using
the user's rebuilt AMGX library. This is a scalar, single-GPU qualification
with FGMRES/NOSOLVER. Matrix upload still copies. BSR integration is paused.
Mixed-precision solves are an accepted existing solver limitation, outside this
milestone. Their regression gate checks error propagation and buffer lifetime.

## Binaries and environment

- AMGX source base: `7096fa209038c8c7a07f1a761af976e6629d59a3`, plus working changes.
- PyAMGX source base: `c1977c51a7e10e5b1b932d39d9d34baed3f63a38`, plus working changes.
- Library: `libamgxsh.so`.
  SHA256: `29f5320e1d26608208a317c30555c64cbbefbc64602129f97681d91bf6bed5b3`.
- Tested extension: `pyamgx.cpython-312-x86_64-linux-gnu.so`.
  SHA256: `aa44c735d6c6931897e9f4ddd722687c9a113bc8a48f09ffdb5fca82f0de636f`.
- Python 3.12.3, CuPy 14.2.0, NVIDIA RTX PRO 5000 Blackwell.
- AMGX reports compiled/loaded CUDA 13.0; CuPy runtime query returns 13020,
  driver query 13000. These are recorded separately, not assumed identical.
- AMGX's embedded build-date string still says September 11 because that
  translation unit was not rebuilt. File timestamp and exported attachment
  symbols confirm the September 28 library; `/proc/self/maps` confirms its use.

## Results

| Check | Result |
| --- | --- |
| Native `BorrowedVectors` | 3 pass: dDDI/dFFI solves and dDFI expected-error/buffer checks |
| Protocol, attachment, existing vector/matrix/solver Python tests | 100 pass, including 3 dDFI error/recovery cases |
| Attachment tests with `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM=1` | 23 pass, including dDFI error/recovery |
| FP64 independent residual, 2-by-2 example | `2.220446049250313e-16` |
| FP32 independent residual, 2-by-2 example | `1.1920928955078125e-07` |

The passing attachment suite includes native/CuPy pointer equality, repeated
RHS writes, nonzero guesses, legacy/PTDS/nonblocking producer streams, independent
consumer streams, in-place zeroing, detach/rebind, owner retention, cyclic GC,
CUDA Array Interface v2 with explicit stream, descriptor changes, overlapping
vectors, empty arrays, and host-pointer rejection. Wrong-device checks are
conditional and were not exercised on this single-GPU machine.

The native launcher does not include the older `VectorTests` source in its
CMake target: requesting it ran zero tests and is not counted as validation.
The new native test does check owning copies, integer comparisons, owning
allocation swaps, and rejection of borrowed resizes/swaps.

## Transfer evidence

Nsight Systems 2025.3.2 captured CUDA and NVTX with all CUDA APIs enabled.
Memcpy records were joined to runtime calls by correlation ID and attributed
to the calling thread's enclosing NVTX range; GPU completion timestamps alone
were not used to assign operations to ranges.

| Range | FP64 | FP32 |
| --- | --- | --- |
| `borrowed.attach` | 0 copies | 0 copies |
| `borrowed.detach` | 0 copies | 0 copies |
| `copy.upload` positive control | 2 D2D copies, 16 bytes each | 2 D2D copies, 8 bytes each |
| `copy.download` positive control | 1 D2D copy, 16 bytes | 1 D2D copy, 8 bytes |
| `borrowed.solve` | 10 scalar D2H copies, 8 bytes each | 10 scalar D2H copies, 4 bytes each |

The solve still transfers scalar reduction results to the host. The verified
claim is shared RHS/solution allocations and no vector boundary staging,
not an algorithm with no internal copies, allocations, or synchronization.
Both examples assert native pointer equality before and after solving.

Historical artifact names (not bundled; `results/` is a generic example
directory) are `results/amgx-borrowed-native-final.log`,
`results/pyamgx-borrowed-all.log`, `results/pyamgx-borrowed-ptds-all.log`, and
`results/pyamgx-borrowed-profile.*` / `results/pyamgx-borrowed-fp32-profile.*`.
The profile families contain the `.nsys-rep`, `.sqlite`, and numerical `.json`
reports. Transfer audit summaries are `results/pyamgx-borrowed-transfer-audit.json`
and `results/pyamgx-borrowed-fp32-transfer-audit.json`.

The original full Python run had three dDFI solve failures.
The AMGX implementation explicitly rejects mixed SpMV at
`src/amgx_cusparse.cu:1344`. No mixed-precision success is claimed, and the
user accepted documenting this limitation instead of adding mixed SpMV support.
The native regression now expects AMGX_ERR_NOT_IMPLEMENTED for dDFI while
retaining its pointer and RHS checks. The user rebuilt the test launcher at
12:09 on September 28, and its final run reports 3 tests, 0 failures.
The AMGX library implementation and recorded hash are unchanged.

All gates for the accepted scalar borrowed-vector milestone are complete:
native storage/lifetime checks, Python interface/error checks, producer and
consumer stream ordering, independent residuals, and transfer profiling.
Mixed-precision solver implementation, matrix borrowing, BSR integration,
and application integration are outside this completed milestone.

A separate two-row diagnostic confirmed that ordinary uploaded dDFI vectors
and attached dDFI vectors return the identical Python error, `Configuration
feature is not implemented.` The same native rejection is present at the AMGX
base revision, before these changes. After the failed borrowed solve, native
pointers still match CuPy, the RHS is unchanged, and detach, reattach, and
in-place zeroing succeed. The process exits cleanly. Evidence is recorded in
`results/pyamgx-mixed-storage-parity.json` and its accompanying `.log`.
