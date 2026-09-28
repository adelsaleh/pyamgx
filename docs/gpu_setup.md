# Build and run GPU examples

Use a Python environment with the dependencies described in
[installation](install.rst), plus CuPy for your CUDA version and pytest for tests.
Build the matching AMGX fork first; see its README. These Linux commands use
placeholder directories: replace every `/path/to/...` with your own location.
No application-specific wrapper script is required.

```bash
cd /path/to/pyamgx
export AMGX_DIR=/path/to/AMGX
export AMGX_BUILD_DIR="$AMGX_DIR/build"
export CUDA_ROOT=/path/to/cuda
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$AMGX_BUILD_DIR:$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Build only the Python extension against the existing AMGX library.
python setup.py build_ext \
  --build-lib "$PWD/build/python" --build-temp "$PWD/build/temp"
export PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results

python examples/reusable_solver.py
```

`AMGX_BUILD_DIR` must contain the matching `libamgxsh.so`; adjust the library
search path if your build puts it in a subdirectory. Use the same CUDA toolkit
and libraries selected for your AMGX build. Rebuild the Python extension when
the native ABI or linked library changes. If using an installed extension
instead, omit the isolated build and its `PYTHONPATH` override.

Commands in the GPU guides run from the PyAMGX repository root with this
environment configured. Save new reports under `results/` or another directory
of your choosing. Published benchmark records identify binaries by filename;
absolute machine paths are omitted. Historical measurements retain their
original values; the example directories are not claims about where they ran.
