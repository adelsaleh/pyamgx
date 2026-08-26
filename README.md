# pyamgx: Python interface to NVIDIA's [AMGX](https://github.com/NVIDIA/AMGX) library

[![Documentation Status](http://readthedocs.org/projects/pyamgx/badge/?version=latest)](http://pyamgx.readthedocs.io/en/latest/?badge=latest)

For installation instructions, overview and examples, see the
[documentation](https://pyamgx.readthedocs.io).

## HDG workspace integration

The `quality-of-life` branch is the PyAMGX binding component qualified with the
sibling HDG solver stack. At commit `6b26b12`, it combines the const-compatible
AMGX print callback with the memory-statistics and error-code bindings from
`73e4af0`. Pair it with
[`adelsaleh/AMGX@hdg-cuda13-integration`](https://github.com/adelsaleh/AMGX/tree/hdg-cuda13-integration).

Rebuild PyAMGX whenever the selected AMGX shared library, its ABI, or its linked
CUDA toolkit changes. The upstream `main` branch does not expose the same
