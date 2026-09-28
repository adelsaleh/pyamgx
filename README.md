# pyamgx: Python interface to NVIDIA's [AMGX](https://github.com/NVIDIA/AMGX) library

[![Documentation Status](http://readthedocs.org/projects/pyamgx/badge/?version=latest)](http://pyamgx.readthedocs.io/en/latest/?badge=latest)

For installation instructions, overview and examples, see the
[documentation](https://pyamgx.readthedocs.io).

## Features in this fork

The `quality-of-life` branch extends `main` and pairs with
[AMGX's `hdg-cuda13-integration` branch](https://github.com/adelsaleh/AMGX/tree/hdg-cuda13-integration).

- **Direct GPU input/output:** `Matrix.attach_CSR(csr)` and `Vector.attach(array)`
  borrow CuPyX/CuPy buffers without staging copies; the solution stays in the
  supplied GPU array. See the [CSR attachment contract](docs/borrowed_csr.md).
- **ReusableSolver:** pass CSR and vectors directly, reuse handles and
  preconditioner setup across RHS changes, and clean up with a context manager
  or idempotent `destroy()`. Call `setup(csr)` again after coefficient changes.
- **Lifetime and stream handling:** retain producer references, validate CUDA
  array-interface buffers, and synchronize producer work before synchronous
  setup/solve calls.
- **Diagnostics:** expose AMGX live/reserved/peak device-memory counters,
  native error codes, and attached-buffer pointers.

With `config`, a CuPyX `csr`, and matching CuPy `rhs`/`solution` arrays prepared:

```python
pyamgx.initialize()
try:
    with pyamgx.ReusableSolver(config, mode="dDDI") as solver:
        solver.setup(csr)
        solver.solve(rhs, out=solution)
        # Update rhs in place, then solve again without repeating setup.
finally:
    pyamgx.finalize()
```

The borrowed-CSR/reusable interface currently supports single-GPU square scalar
CSR, int32 indices, FP32 (`dFFI`) or FP64 (`dDDI`), and an explicit diagonal in
every row. CSR must be canonical; keep its structure fixed while attached.
AMGX still allocates internal workspaces and coarse levels. Existing upload and
download APIs remain available. Build the extension against the matching AMGX
headers/library; rebuild it when that library's ABI or CUDA linkage changes.

See [portable GPU setup](docs/gpu_setup.md), the [complete example](examples/reusable_solver.py),
[API and limitations](docs/reusable_solver.md), and
[million-DOF AMG reuse/reconstruction benchmark](docs/reusable_solver_amg_million.md).
