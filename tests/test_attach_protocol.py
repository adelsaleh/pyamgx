"""CPU-only validation: does not import the compiled PyAMGX extension."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "attach_protocol", Path(__file__).parents[1] / "pyamgx/cuda_array_interface.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


class Producer:
    def __init__(self, **changes):
        self.desc = dict(version=3, shape=(3,), typestr="<f8", data=(4096, False),
                         strides=None, stream=1)
        self.desc.update(changes)
        self.reads = 0

    @property
    def __cuda_array_interface__(self):
        self.reads += 1
        return self.desc


@pytest.mark.parametrize("stream", [1, 2, 4096, None])
def test_reads_once_and_preserves_stream(stream):
    producer = Producer(stream=stream)
    assert protocol._attached_descriptor(producer, np.float64) == (4096, 3, 24, stream or 0)
    assert producer.reads == 1


def test_v2_requires_explicit_stream():
    producer = Producer(version=2)
    with pytest.raises(ValueError, match="explicit producer stream"):
        protocol._attached_descriptor(producer, np.float64)
    assert protocol._attached_descriptor(producer, np.float64, 2)[-1] == 2


@pytest.mark.parametrize("changes", [
    {"version": 1}, {"version": 4}, {"version": True}, {"version": 3.0},
    {"shape": (2, 3)}, {"shape": ()}, {"shape": [3]}, {"shape": (-1,)},
    {"shape": (True,)}, {"shape": (3.0,)}, {"shape": (2**31,)},
    {"typestr": ">f8"}, {"typestr": "<f4"}, {"typestr": "<i8"},
    {"typestr": "|V8"}, {"typestr": "bad"}, {"mask": object()},
    {"descr": [("value", "<f8")]}, {"strides": (16,)}, {"strides": (0,)},
    {"strides": (-8,)}, {"strides": (8, 8)}, {"strides": (8.0,)},
    {"data": (4096, True)}, {"data": (0, False)}, {"data": (-8, False)},
    {"data": (4097, False)}, {"data": (2**64, False)}, {"data": (True, False)},
    {"data": (4096, 0)}, {"data": [4096, False]}, {"data": (4096,)},
    {"stream": 0}, {"stream": -1}, {"stream": True}, {"stream": 1.0},
])
def test_rejects_unsafe_descriptors(changes):
    with pytest.raises((ValueError, TypeError)):
        protocol._attached_descriptor(Producer(**changes), np.float64)


def test_empty_null_and_singleton_strides():
    assert protocol._attached_descriptor(
        Producer(shape=(0,), data=(0, False)), np.float64)[:3] == (0, 0, 0)
    assert protocol._attached_descriptor(Producer(shape=(1,), strides=(32,)), np.float64)[1] == 1


def test_float32_and_unstructured_descr():
    producer = Producer(typestr="<f4", strides=(4,), descr=[("", "<f4")])
    assert protocol._attached_descriptor(producer, np.float32)[:3] == (4096, 3, 12)


def test_host_array_is_rejected():
    with pytest.raises(TypeError, match="CUDA array"):
        protocol._attached_descriptor(np.zeros(3), np.float64)
