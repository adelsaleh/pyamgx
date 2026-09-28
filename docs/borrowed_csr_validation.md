# Borrowed CSR validation

The CuPyX CSR milestone is qualified for the single-GPU scalar CSR contract in
[the usage guide](borrowed_csr.md). Native regressions, Python integration,
memory checking, and FP32/FP64 transfer checks pass.

## Results

| Check | Result |
| --- | --- |
| CPU vector/CSR descriptor tests | 60 passed |
| CSR + vector + copying API + protocol Python suite | 136 passed |
| CSR/vector attachment with CuPy PTDS enabled | 43 passed |
| Strengthened canonical-but-missing-diagonal rejection | 1 passed |
| Native BorrowedVectors | 3 passed, 0 failures |
| FP64 small CSR solve under Compute Sanitizer memcheck | 0 errors |
| Native BorrowedCSR | 3 passed, 0 failures |

Python coverage includes FP32/FP64 FGMRES, Block Jacobi and classical AMG
preconditioning, unchanged fine CSR buffers, repeated in-place value updates
followed by setup, native/CuPy pointer identity, legacy/PTDS/nonblocking producer
streams, independent consumer streams, malformed structure rejection, buffer
replacement detection, retained owners, GC cleanup, forbidden mutation, and failed
setup cleanup. dDFI checks the existing unsupported mixed-precision SpMV error.
Only one GPU was available; multi-GPU/device-mismatch qualification is not claimed.

## Transfer evidence

Nsight Systems CUDA/NVTX profiles cover `examples/borrowed_csr.py` in both modes,
with FGMRES/NOSOLVER and the copying control enabled. Native pointers match all
three CuPy buffer pointers before and after setup/solve. Independent residuals:

- dDDI: `2.5121479338940403e-15`.
- dFFI: `2.1324806311895372e-06`.

| NVTX range | FP64 transfers | FP32 transfers |
| --- | --- | --- |
| csr.attach | one 4-byte D2H status; one 4-byte D2D metadata copy | same |
| csr.setup | none | none |
| csr.solve | fifteen 8-byte D2H scalar reductions | fifteen 4-byte D2H scalar reductions |
| csr.detach | none | none |
| csr.copy_upload | full CSR D2D copies: 16, 28, 56 bytes, plus scalar initialization | full CSR D2D copies: 16, 28, 28 bytes, plus scalar initialization |

There is no staging copy of the supplied fine CSR arrays in the borrowed path.
This is not a claim of zero internal copies, metadata, workspaces, scalar
transfers, or synchronization. Coarse AMG allocations are separate from the
shared fine operator. The transfer profile uses NOSOLVER; AMG numerical and
storage-preservation evidence comes from the integration tests and the native
finest-level `setA` reference path.

Range membership uses CUDA runtime API start time and host thread, joined to GPU
copy events by correlation ID. GPU completion timestamps alone are not used to
assign asynchronous copies to ranges. The positive copying control confirms
that full array transfers are visible in the trace.

## Environment and artifacts

- NVIDIA RTX PRO 5000 Blackwell, one GPU; CuPy 14.2.0.
- AMGX built with CUDA 13.0; CuPy runtime query 13020; driver query 13000.
- Loaded library verified through `/proc/self/maps`:
  `libamgxsh.so`.
- Library SHA256: `b3453ec2d341094c3ffcb7ce4929c8fb8b8fedfc04b62497614dc5e194f5a744`.
- Native test launcher SHA256: `5ff1bfbb84b6fb01cc081ac33f3fdeeffabbe6edc0a4b3171dfb31e56cf5064b`.
- Isolated extension: `pyamgx.cpython-312-x86_64-linux-gnu.so`.
- Extension SHA256: `fa3852ed955c31f18d29f19afcd438703f96fb0d4e6b7243656511413485859a`.
- The installed Python extension has not been replaced.

Historical artifact names (raw logs/profiles are not bundled; `results/` is a
generic example directory):

- `results/pyamgx-csr-all-tests.log`, `results/pyamgx-csr-ptds.log`.
- `results/pyamgx-csr-missing-diagonal.log`, `results/amgx-csr-vector-regression.log`.
- `results/amgx-borrowed-csr-native-final.log`.
- `results/pyamgx-csr-memcheck.log`, `results/pyamgx-csr-runtime.json`.
- `results/pyamgx-csr-fp64-profile.{nsys-rep,sqlite,json,log}` and the corresponding
  `fp32` files.
- `results/pyamgx-csr-transfer-audit.json` contains range counts and binary hashes.

## Native regression qualification

The test launcher was rebuilt on 2026-09-28 against the same library used
for the Python and transfer checks; its hash is unchanged. Running `amgx_tests_launcher BorrowedCSR` with the matching CUDA libraries exits successfully with three tests and zero failures.

The native fixture covers pointer identity, rejection of copying APIs and
premature detach/destroy, solve, detach/reattach, and preservation of caller-owned
buffers after matrix destruction. Its allocations are destroyed before their
resource memory pool. Both dDDI and dFFI solve successfully; dDFI passes by
returning the expected `AMGX_RC_NOT_IMPLEMENTED` mixed-precision SpMV error.

To rebuild the native fixture, the user runs:

```sh
cmake --build /path/to/AMGX/build \
  --target amgx_tests_launcher --parallel 16
```

Then rerun `amgx_tests_launcher BorrowedCSR` with the matching CUDA libraries.
