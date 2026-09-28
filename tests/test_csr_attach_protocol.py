"""CPU-only CSR protocol checks; no AMGX shared library required."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_attach_protocol import Producer, protocol


def csr():
    return SimpleNamespace(format="csr", shape=(2, 2),
        indptr=Producer(shape=(3,), typestr="<i4", data=(4096, False)),
        indices=Producer(shape=(4,), typestr="<i4", data=(8192, False)),
        data=Producer(shape=(4,), typestr="<f8", data=(12288, False)))


def test_keeps_three_original_buffers_and_streams():
    a = csr()
    a.indices.desc["stream"] = 2
    a.data.desc["stream"] = 12345
    shape, arrays, desc = protocol._attached_csr_descriptor(a, np.float64)
    assert shape == (2, 2)
    assert arrays == (a.indptr, a.indices, a.data)
    assert [d[0] for d in desc] == [4096, 8192, 12288]
    assert [d[3] for d in desc] == [1, 2, 12345]
    assert all(array.reads == 1 for array in arrays)


@pytest.mark.parametrize("field,value", [
    ("format", "csc"), ("shape", (2, 3)), ("shape", (0, 0)),
    ("shape", (True, True)), ("shape", (2.0, 2.0)),
    ("shape", (2**31 - 1, 2**31 - 1)), ("shape", (2,)),
])
def test_rejects_incompatible_container(field, value):
    a = csr()
    setattr(a, field, value)
    with pytest.raises((ValueError, TypeError)):
        protocol._attached_csr_descriptor(a, np.float64)


@pytest.mark.parametrize("buffer,change", [
    ("indptr", {"shape": (2,)}), ("indices", {"shape": (3,)}),
    ("indices", {"typestr": "<i8"}), ("data", {"typestr": "<f4"}),
    ("data", {"strides": (16,)}), ("data", {"data": (4096, False)}),
    ("indices", {"data": (8192, True)}),
])
def test_rejects_incompatible_buffer(buffer, change):
    a = csr()
    getattr(a, buffer).desc.update(change)
    with pytest.raises(ValueError):
        protocol._attached_csr_descriptor(a, np.float64)


def test_v2_override_covers_each_buffer():
    a = csr()
    for array in (a.indptr, a.indices, a.data):
        array.desc["version"] = 2
    with pytest.raises(ValueError, match="v2"):
        protocol._attached_csr_descriptor(a, np.float64)
    _, _, desc = protocol._attached_csr_descriptor(a, np.float64, stream=2)
    assert all(d[3] == 2 for d in desc)
