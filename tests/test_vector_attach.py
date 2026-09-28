"""Small GPU regressions; requires the rebuilt borrowed-vector AMGX/PyAMGX pair."""
import gc
import weakref

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
import pyamgx


def queue_delay(stream):
    # Leave producer writes behind a bounded kernel so ordering tests do not
    # pass merely because a tiny fill completed before the native call began.
    kernel = cp.RawKernel(r'''
    extern "C" __global__ void delay() {
        unsigned long long start = clock64();
        while (clock64() - start < 20000000ULL) {}
    }
    ''', "delay")
    with stream:
        kernel((1,), (1,), ())


@pytest.fixture
def handles():
    assert hasattr(pyamgx.Vector, "attach"), "Rebuild PyAMGX against the borrowed-vector AMGX build"
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    pyamgx.initialize()
    objects = []
    def own(obj):
        objects.append(obj)
        return obj
    try:
        cfg = own(pyamgx.Config().create_from_dict({
            "config_version": 2, "exception_handling": 0,
            "solver": {"solver": "FGMRES", "preconditioner": {"solver": "NOSOLVER"},
                       "max_iters": 30, "gmres_n_restart": 8, "tolerance": 1e-6,
                       "monitor_residual": 1, "convergence": "RELATIVE_INI_CORE"}}))
        resources = own(pyamgx.Resources().create_simple(cfg))
        yield own, cfg, resources
    finally:
        for obj in reversed(objects):
            obj.destroy()
        pyamgx.finalize()


@pytest.mark.parametrize("mode,dtype", [("dDDI", np.float64), ("dFFI", np.float32)])
def test_pointer_identity_inplace_fill_and_detach(handles, mode, dtype):
    own, _, resources = handles
    array = cp.full(7, 3, dtype=dtype)
    vector = own(pyamgx.Vector().create(resources, mode=mode)).attach(array)
    assert vector.is_attached and vector.attached_ptr == array.data.ptr
    vector.set_zero()
    # No consumer synchronization with AMGX is supplied by the caller.
    with cp.cuda.Stream(non_blocking=True):
        assert float(cp.sum(cp.abs(array)).get()) == 0
    with pytest.raises(pyamgx.AMGXError):
        vector.set_zero(8, 1)
    assert vector.attached_ptr == array.data.ptr
    with pytest.raises(RuntimeError, match="detach"):
        vector.upload(array)
    assert vector.detach() is array
    assert not vector.is_attached and vector.get_size() == (0, 1)
    vector.attach(array)


def test_owner_is_retained_until_detach(handles):
    own, _, resources = handles
    array = cp.ones(3)
    ref = weakref.ref(array)
    vector = own(pyamgx.Vector().create(resources)).attach(array)
    del array
    gc.collect()
    assert ref() is not None
    vector.set_zero()
    vector.detach()
    gc.collect()
    assert ref() is None


def test_implicit_vector_destruction_releases_borrow(handles):
    _, _, resources = handles
    array = cp.ones(3)
    ref = weakref.ref(array)
    vector = pyamgx.Vector().create(resources).attach(array)
    del array, vector
    gc.collect()
    assert ref() is None


def test_cyclic_producer_lifetime(handles):
    _, _, resources = handles
    class Producer:
        def __init__(self):
            self.array = cp.ones(3)
        @property
        def __cuda_array_interface__(self):
            return self.array.__cuda_array_interface__
    producer = Producer()
    ref = weakref.ref(producer.array)
    producer.vector = pyamgx.Vector().create(resources).attach(producer)
    del producer
    gc.collect()
    assert ref() is None


def test_v2_with_explicit_stream(handles):
    own, _, resources = handles
    array = cp.ones(3)
    class Producer:
        @property
        def __cuda_array_interface__(self):
            desc = dict(array.__cuda_array_interface__)
            desc["version"] = 2
            desc.pop("stream", None)
            return desc
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        queue_delay(stream)
        array.fill(7)
        vector = own(pyamgx.Vector().create(resources)).attach(Producer(), stream=stream.ptr)
        vector.set_zero()
    with cp.cuda.Stream(non_blocking=True):
        assert float(cp.sum(array).get()) == 0


@pytest.mark.parametrize("mode,dtype", [("dFFI", np.float32), ("dDDI", np.float64)])
def test_owning_transfer_precision_is_preserved(handles, mode, dtype):
    own, _, resources = handles
    vector = own(pyamgx.Vector().create(resources, mode=mode))
    values = np.array([1., 2., 3.], dtype=dtype)
    vector.upload(values)
    result = vector.download()
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, values)
    with pytest.raises(pyamgx.AMGXError):
        vector.attach(cp.ones(3, dtype=dtype))
    np.testing.assert_array_equal(vector.download(), values)


@pytest.mark.parametrize("mode,dtype", [("dDDI", np.float64), ("dFFI", np.float32), ("dDFI", np.float64)])
@pytest.mark.parametrize("kind", ["legacy", "ptds", "nonblocking"])
def test_solve_storage_and_stream_contract(handles, mode, dtype, kind):
    own, cfg, resources = handles
    # Matrix upload is unchanged by this vector-only milestone.
    matrix = own(pyamgx.Matrix().create(resources, mode=mode))
    values = np.array([4., 1., 1., 3.], dtype=np.float32 if mode[2] == 'F' else np.float64)
    matrix.upload(np.array([0, 2, 4], np.int32), np.array([0, 1, 0, 1], np.int32),
                  values, shape=(2, 2))
    solver = own(pyamgx.Solver().create(resources, cfg, mode=mode))
    solver.setup(matrix)
    stream = {"legacy": cp.cuda.Stream.null, "ptds": cp.cuda.Stream.ptds}.get(kind)
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        queue_delay(stream)
        b_array = cp.array([1., 2.], dtype=dtype)
        x_array = cp.full(2, 0.125, dtype=dtype)
        b = own(pyamgx.Vector().create(resources, mode=mode)).attach(b_array)
        x = own(pyamgx.Vector().create(resources, mode=mode)).attach(x_array)
        if mode == "dDFI":
            # Existing AMGX mixed SpMV limitation, independent of borrowing.
            with pytest.raises(pyamgx.AMGXError, match="not implemented"):
                solver.solve(b, x)
            assert b.attached_ptr == b_array.data.ptr
            assert x.attached_ptr == x_array.data.ptr
            cp.testing.assert_array_equal(b_array, cp.array([1., 2.]))
            assert b.detach() is b_array and x.detach() is x_array
            b.attach(b_array)
            x.attach(x_array).set_zero()
            cp.testing.assert_array_equal(x_array, cp.zeros(2))
            uploaded_b = own(pyamgx.Vector().create(resources, mode=mode)).upload(np.array([1., 2.]))
            uploaded_x = own(pyamgx.Vector().create(resources, mode=mode)).upload(np.full(2, 0.125))
            with pytest.raises(pyamgx.AMGXError, match="not implemented"):
                solver.solve(uploaded_b, uploaded_x)
            return
        for multiplier in (1., 2., 3.):
            queue_delay(stream)
            b_array[:] = cp.array([multiplier, 2 * multiplier], dtype=dtype)
            x_array.fill(0.125)
            solver.solve(b, x)
            assert b.attached_ptr == b_array.data.ptr
            assert x.attached_ptr == x_array.data.ptr
            with cp.cuda.Stream(non_blocking=True):
                expected = cp.array([multiplier / 11, 7 * multiplier / 11], dtype=dtype)
                cp.testing.assert_allclose(x_array, expected, rtol=2e-5, atol=2e-6)
                residual = cp.array([[4., 1.], [1., 3.]], dtype=dtype) @ x_array - b_array
                assert float(cp.linalg.norm(residual).get()) < 2e-5


@pytest.mark.parametrize("kind", ["slice", "readonly", "changed"])
def test_rejects_unsafe_array_storage(handles, kind):
    own, _, resources = handles
    array = cp.ones(6)
    vector = own(pyamgx.Vector().create(resources))
    if kind == "slice":
        with pytest.raises(ValueError, match="contiguous"):
            vector.attach(array[::2])
        return
    class Producer:
        @property
        def __cuda_array_interface__(self):
            desc = dict(array.__cuda_array_interface__)
            if kind == "readonly":
                desc["data"] = (array.data.ptr, True)
            elif self.changed:
                desc["shape"] = (3,)
            return desc
        changed = False
    producer = Producer()
    if kind == "readonly":
        with pytest.raises(ValueError, match="writable"):
            vector.attach(producer)
    else:
        vector.attach(producer)
        producer.changed = True
        with pytest.raises(ValueError, match="storage changed"):
            vector.set_zero()


def test_attached_rhs_solution_alias_is_rejected(handles):
    own, cfg, resources = handles
    matrix = own(pyamgx.Matrix().create(resources))
    matrix.upload(np.array([0, 1, 2], np.int32), np.array([0, 1], np.int32),
                  np.array([2., 3.]), shape=(2, 2))
    solver = own(pyamgx.Solver().create(resources, cfg))
    solver.setup(matrix)
    array = cp.ones(3)
    b = own(pyamgx.Vector().create(resources)).attach(array[:2])
    x = own(pyamgx.Vector().create(resources)).attach(array[1:])
    with pytest.raises(pyamgx.AMGXError):
        solver.solve(b, x)
    cp.testing.assert_array_equal(array, cp.ones(3))


def test_wrong_device_and_host_pointer_rejected(handles):
    own, _, resources = handles
    vector = own(pyamgx.Vector().create(resources))
    host = np.ones(3)
    class HostDisguisedAsCUDA:
        __cuda_array_interface__ = dict(host.__array_interface__, version=3, stream=None)
    with pytest.raises(pyamgx.AMGXError):
        vector.attach(HostDisguisedAsCUDA())
    assert not vector.is_attached
    if cp.cuda.runtime.getDeviceCount() > 1:
        with cp.cuda.Device(1):
            array = cp.ones(3)
            with pytest.raises(pyamgx.AMGXError):
                vector.attach(array)


def test_empty_repeated_attach_and_destroy_state(handles):
    own, _, resources = handles
    vector = own(pyamgx.Vector().create(resources))
    empty = cp.empty(0)
    vector.attach(empty)
    assert vector.get_size() == (0, 1)
    with pytest.raises(RuntimeError, match="already attached"):
        vector.attach(empty)
    assert vector.detach() is empty
    standalone = pyamgx.Vector().create(resources).attach(cp.ones(3))
    standalone.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        standalone.attach(cp.ones(3))
