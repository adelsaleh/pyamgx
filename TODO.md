# pyamgx communication, logging, and CuPy roadmap

This roadmap makes pyamgx a reliable Python boundary for AMGX: numerical telemetry must be correct, C failures must become useful Python exceptions, logs must integrate with Python tooling, and NumPy/CuPy transfers must have explicit device and stream semantics. Implementation and tests should use the local `../AMGX` and `../cupy` repositories as the authoritative C API and CuPy behavior references.

Work for this roadmap continues on local development branch `quality-of-life`, matching the AMGX development branch.

## P0 — expose correct solve results

- [ ] Add a Python regression for the upstream AMGX BiCGSTAB residual bug.
  - On the 3-by-3 nonsymmetric reproduction run on 2026-08-11, AMGX printed and `Solver.get_residual(1)` returned `3.7416573867739413`, while NumPy recomputation from the downloaded solution gave `||b-Ax||_2 = 0.7197162587140464` after one iteration.
  - Assert console/callback output, `get_residual(i)`, final status, iteration count, and an independent NumPy or CuPy residual agree after the AMGX fix.
  - Parameterize normal and early-exit BiCGSTAB paths, preconditioning, nonzero initial guesses, float/double modes, and more than one iteration.
- [ ] Correct `AMGX_SOLVE_NOT_CONVERGED` in `pyamgx/Solver.pyx` from `2` to the AMGX header value `3`; use value equality consistently and test all four statuses (`success`, `failed`, `diverged`, `not_converged`). Prefer generating/importing enum declarations from `amgx_c.h` so bindings cannot drift silently.
- [ ] Bind `AMGX_solver_calculate_residual_norm` and offer an explicit `Solver.calculate_residual_norm(A, b, x)` API for validating the current solution independently of stored history.
- [ ] Add `Solver.residual_history(...)` returning a NumPy array (and optionally a CuPy array) with documented iteration indexing, block layout, norm, and estimated/true-residual semantics.

## P0 — error and callback correctness

- [ ] Check every AMGX return code. Known omissions include callback registration, `get_api_version`, and `read_system`; add a source/test audit that prevents unchecked `AMGX_RC` results from recurring.
- [ ] Replace the global, implicitly initialized `print_callback` with managed callback state.
  - Validate callability, keep a strong reference for the registration lifetime, support unregister/reset, and restore the prior handler from a context manager.
  - Decode exactly the callback’s `(msg, length)` bytes with a documented error policy instead of assuming NUL termination or ignoring `length`.
  - Acquire the GIL correctly if AMGX invokes from a non-Python thread. Never let a Python exception escape the `noexcept` C callback: save it for a safe Python boundary or route it to `logging`/`sys.unraisablehook` with context.
  - Document process-global scope, thread safety, reentrancy, shutdown/finalization ordering, and behavior after `fork`.
- [ ] Enrich `AMGXError` with `code`, symbolic name, operation, and chained context while preserving a useful `str(exc)`. Test initialization/configuration/CUDA/setup/solve failures.
- [ ] Make initialization/finalization idempotence and ownership explicit (reference-counted context manager or a clearly enforced single owner); avoid deprecated plugin calls when the linked AMGX version no longer requires them.

## P1 — Python logging integration

- [ ] Add `pyamgx.install_logger(logger=None, level=..., rank=...)` and `pyamgx.capture_output(...)` APIs built on callback management; default to a named `logging.getLogger("pyamgx")` without configuring application handlers.
- [ ] Preserve AMGX record boundaries and map structured AMGX severities/components when the upstream structured callback becomes available. Until then, keep text intact and use conservative severity mapping rather than fragile parsing of residual tables.
- [ ] Attach solver/scope, rank/device, and iteration metadata via `LogRecord.extra` when available. Prevent duplicate handlers/messages across repeated initialize/finalize cycles.
- [ ] Test custom handlers, Unicode/non-NUL-terminated messages, callback exceptions, callback replacement/restoration, multi-threaded use, MPI rank filtering, and `caplog` capture.
- [ ] Document recipes for quiet library use, console progress, file logging, Jupyter, MPI, and collecting a support bundle with pyamgx/AMGX/CUDA/GPU versions.

## P1 — reusable solver and preconditioner lifecycle

- [ ] Add a first-class `ReusableSolver` context manager for fixed operators.
  - Own `Config`, `Resources`, `Matrix`, reusable RHS/solution vectors, and `Solver` with deterministic cleanup; expose `setup(A)` once followed by repeated `solve(b, x0=None)` calls without recreating handles.
  - Publish observable state such as `is_setup`, shape/block dimensions, dtype/mode, device, setup generation, solve count, last setup action, and whether the hierarchy was rebuilt or reused.
  - Reject use-after-close, solve-before-setup, incompatible vectors, and configuration mutation after setup with precise Python exceptions.
- [ ] Make matrix updates safe and convenient.
  - Provide `replace_coefficients(..., resetup="auto|reuse_structure|rebuild")` with documented mappings to AMGX setup/resetup policies.
  - Track the bound matrix and its revision so a stale hierarchy cannot be used silently after values, sparsity, block layout, precision, scaling, device, or relevant solver/preconditioner configuration changes.
  - Keep an explicit expert escape hatch, but make the safe automatic policy the default.
- [ ] Minimize overhead in steady-state solves.
  - Reuse AMGX vector handles and capacity; avoid allocating temporary NumPy/CuPy arrays or rebuilding Python descriptors when shape, dtype, and device are unchanged.
  - Release the GIL around safe AMGX setup/solve calls, reduce Python/Cython transitions, and expose synchronous and future stream-aware paths without hidden device-wide synchronization.
  - Return structured per-call results containing setup action/time, solve time, iterations, status/reason, residuals, and transfer/synchronization timings rather than requiring log parsing.
- [ ] Add `solve_many` for repeated right-hand sides.
  - Accept stacked NumPy/CuPy arrays or an iterable with explicit layout and output ownership, reuse vector/workspace capacity, and call an AMGX batched API when available.
  - Until AMGX provides batching, implement a low-overhead loop that retains all handles and clearly distinguishes transfer, solve, and validation costs.
- [ ] Add reuse-focused tests, examples, and benchmarks.
  - Demonstrate SciPy CSR and CuPyX CSR setup once followed by many right-hand sides, coefficient-only resetup, and automatic full invalidation on structural changes.
  - Verify independently computed residuals and native/Python timing agreement; assert the second and later solves report zero setup work when unchanged.
  - Benchmark cold construction, warm solve latency, Python overhead, transfers, allocations, and memory stability over hundreds of solves against direct low-level pyamgx usage.

## P1 — CuPy integration without host staging

- [ ] Complete [pyamgx issue #26](https://github.com/shwina/pyamgx/issues/26): make `Vector.download()` accept writable contiguous NumPy arrays and CUDA-array-interface objects such as `cupy.ndarray`; keep `download_raw` as an advanced escape hatch.
- [ ] Implement one internal buffer-descriptor parser for `__array_interface__` and `__cuda_array_interface__`, used by `Matrix.upload`, `Matrix.replace_coefficients`, `Vector.upload`, and `Vector.download`.
  - Read each protocol property once and validate `version`, `data`, `shape`, `typestr`, `strides`, optional `descr`/`mask`, dimensionality, byte order, size, contiguity, and writability for outputs.
  - Support CUDA Array Interface v2 deliberately and v3 with its `stream` field; reject unknown versions, masks, negative/overlapping strides, null nonempty buffers, and unsupported structured/big-endian dtypes with precise errors.
  - Treat v3 stream values correctly: `1` is the legacy default stream, `2` is the per-thread default stream, `0` is invalid, and other values are `cudaStream_t` handles.
  - Validate pointer device with CUDA pointer attributes, compare it with the AMGX resource device, and reject HIP-backed CuPy arrays because AMGX is CUDA-only.
  - Reject mixed-host/device CSR buffers unless explicitly supported and tested. Keep source/destination objects and any external stream/event wrappers alive until AMGX’s copy completes.
- [ ] Define a strict CuPy/CuPyX sparse contract.
  - Accept `cupyx.scipy.sparse.csr_matrix` and `csr_array` through their `data`, `indices`, and `indptr` device arrays without calling `.get()` or `cupy.asnumpy()`.
  - Require or explicitly convert C-contiguous values and AMGX-supported index widths. Never silently narrow CuPy’s 64-bit indices; range-check any opt-in conversion and perform it on-device.
  - Decide and test semantics for unsorted indices, duplicate entries, `has_sorted_indices`, `has_canonical_format`, empty shapes/rows, explicit zeros, and block matrices. Do not mutate the caller’s sparse object.
  - Keep CuPy optional: protocol-compatible producers should work without importing CuPy, while CuPy-specific convenience helpers and tests use a lazy optional import.
- [ ] Describe transfers accurately: current AMGX upload/download uses `cudaMemcpyDefault` into AMGX-owned storage, enabling direct device-to-device copies but not zero-copy sharing of CuPy allocations.
- [ ] Define CUDA stream semantics.
  - Consume CuPy’s exported CUDA Array Interface v3 `stream` value and establish producer-to-AMGX ordering. Prefer event waits on a stream-aware AMGX API; until that exists, use the narrowest documented producer-stream synchronization and expose that cost.
  - Establish AMGX-to-consumer ordering before a downloaded CuPy array is used. If AMGX exposes an execution stream, wrap a stream-protocol object with `cupy.cuda.Stream.from_external()`; do not build new code on CuPy’s deprecated `ExternalStream`.
  - Consider a pyamgx stream wrapper implementing `__cuda_stream__` only when AMGX can provide a valid stream handle, device ID, and lifetime. Do not fabricate protocol support around the legacy default stream.
  - Exercise CuPy producer modes controlled by `CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION` and `CUPY_CUDA_PER_THREAD_DEFAULT_STREAM`. Test `CUPY_CUDA_ARRAY_INTERFACE_SYNC` only when CuPy consumes a pyamgx/AMGX-exported array; it controls CuPy as a consumer, not pyamgx as a consumer.
  - Do not rely silently on legacy-default-stream behavior or call `cupy.cuda.Device.synchronize()`/global `cudaDeviceSynchronize()` in normal interoperability paths.
- [ ] Add end-to-end CuPy/CuPyX CSR examples and GPU tests: upload matrix/RHS/guess, solve, download into CuPy, and compute `cupy.linalg.norm(b - A @ x)` without host staging. Mark tests by CUDA availability and include NumPy parity tests.
- [ ] Mirror the protocol edge cases exercised by local CuPy tests such as `tests/cupy_tests/core_tests/test_ndarray_cuda_array_interface.py` and `cupy/testing/_protocol_helpers.py`; do not import CuPy’s private test helpers into pyamgx.
- [ ] Run stream tests with work queued before upload and immediately after download so missing dependencies fail deterministically, not just timing-dependent smoke tests. Cover legacy, per-thread-default, and ordinary nonblocking streams, two devices where available, wrong-device arrays, sliced/noncontiguous arrays, zero-length arrays, and early object release.
- [ ] Add an optional integration test job against the local CuPy checkout and a supported released CuPy wheel. Record CuPy version/commit, AMGX version/commit, CUDA runtime/driver, GPU, CUDA Array Interface export version, and PTDS setting in failures.
- [ ] Benchmark NumPy versus CuPy transfer paths and verify with profiling that the CuPy path performs device-to-device transfers and no hidden device-to-host-to-device bounce.

## P1 — AMGX/pyamgx/CuPy contract split

- [ ] Keep responsibilities explicit across repositories.
  - AMGX owns pointer-kind/device validation, copy direction, stream/event execution semantics, completion, numerical status, and a stable C ABI.
  - pyamgx owns Python protocol parsing, Python object lifetimes, exceptions/logging, optional CuPy conveniences, and translating Python stream/device context into AMGX calls.
  - CuPy remains an unmodified reference/producer/consumer unless testing demonstrates a CuPy defect; pyamgx must follow its public CUDA Array Interface and CUDA Stream Protocol rather than depend on private implementation details.
- [ ] Gate asynchronous pyamgx transfer/solve APIs on the corresponding AMGX completion primitives. Synchronous methods remain safe and explicit; an `async_` name must return an event/future or otherwise provide a testable completion boundary.
- [ ] Add a shared interoperability fixture/system specification used by AMGX C tests and pyamgx NumPy/CuPy tests so matrix values, expected residual history, statuses, and stream ordering are comparable across layers.
- [ ] Decide whether AMGX-owned vectors should ever be exposed as borrowed CuPy arrays. Do not implement this until AMGX offers stable pointer access, allocation size/type/device metadata, mutation rules, and ownership/completion hooks; use copies in the initial integration.

## P1 — known integration issues

- [ ] Address [issue #35](https://github.com/shwina/pyamgx/issues/35): expose appropriate resource/device APIs, document AMGX’s process/thread constraints, release the GIL around safe long-running setup/solve calls, and test concurrent solvers before claiming multithreading support.
- [ ] Address [issue #33](https://github.com/shwina/pyamgx/issues/33): make OpenMP linkage/runtime requirements reproducible and test importing the built wheel against supported AMGX releases.
- [ ] Add explicit GPU selection/resource creation to cover [issue #28](https://github.com/shwina/pyamgx/issues/28), with CuPy current-device interoperability tests and clear mismatch errors.
- [ ] Triage the current [pyamgx issue list](https://github.com/shwina/pyamgx/issues) each release. Link accepted issues to tests, record supported AMGX/CUDA/Python/NumPy/CuPy versions, and publish compatibility changes.

## P2 — API, packaging, and documentation

- [ ] Replace legacy `setup.py`-only builds with a supported `pyproject.toml` flow and CI-built artifacts or a clearly documented source-build path. Test clean imports so missing AMGX/CUDA/OpenMP libraries name the missing dependency and remediation.
- [ ] Add context managers and safe lifecycle state for `Config`, `Resources`, `Matrix`, `Vector`, and `Solver`; keep explicit `destroy()` available and make double-destroy/use-after-destroy deterministic Python errors.
- [ ] Publish a diagnostics guide shared with AMGX: residual definitions, history indexing, stop statuses, callback/logging behavior, performance costs, and a minimal BiCGSTAB correctness example.
- [ ] Publish a CuPy interoperability guide with supported dense/sparse types, dtypes/index widths, copy-versus-borrow behavior, device selection, current-stream handling, synchronization costs, object lifetime, and multi-GPU examples.
- [ ] Add CI tiers for CPU-only import/error tests, single-GPU NumPy tests, CuPy tests, and optional multi-GPU/MPI tests, with an AMGX-version matrix.

## Definition of done

- pyamgx never silently drops an AMGX error, and Python callbacks cannot corrupt interpreter state or escape through a `noexcept` boundary.
- Python logs, residual history, independent residual calculation, status, and iteration count agree with AMGX for every tested solver path.
- NumPy and CuPy inputs have documented copy, device, stream, dtype, and lifetime behavior; CuPy round trips require no host staging.
- Non-default-stream tests prove producer → AMGX → consumer ordering without device-wide synchronization, and protocol-compatible CUDA arrays work without a mandatory CuPy dependency.
- Every supported AMGX/CUDA/Python/NumPy/CuPy combination is documented and continuously tested.
