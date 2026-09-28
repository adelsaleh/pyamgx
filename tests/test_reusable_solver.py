"""Borrowing, refresh, recovery and lifetime checks for the convenience API."""
import gc
import weakref

import pytest

cp = pytest.importorskip("cupy")
from cupyx.scipy import sparse
import pyamgx


def config():
    return {
        "config_version": 2, "exception_handling": 0,
        "solver": {"solver": "FGMRES", "preconditioner": {"solver": "NOSOLVER"},
                   "max_iters": 30, "gmres_n_restart": 5,
                   "monitor_residual": 1, "convergence": "RELATIVE_INI_CORE",
                   "tolerance": 1e-6},
    }


def matrix(dtype=cp.float64):
    return sparse.csr_matrix(cp.asarray([[4, 1], [1, 3]], dtype=dtype))


@pytest.fixture(autouse=True)
def initialized():
    pyamgx.initialize()
    yield
    pyamgx.finalize()


@pytest.mark.parametrize("mode,dtype", [("dDDI", cp.float64), ("dFFI", cp.float32)])
@pytest.mark.parametrize("stream_kind", ["legacy", "nonblocking", "ptds"])
def test_repeated_solve_and_shared_pointers(mode, dtype, stream_kind):
    stream = {"legacy": cp.cuda.Stream.null, "ptds": cp.cuda.Stream.ptds,
              "nonblocking": cp.cuda.Stream(non_blocking=True)}[stream_kind]
    with stream, pyamgx.ReusableSolver(config(), mode) as solver:
        csr = matrix(dtype)
        rhs = cp.empty(2, dtype=dtype)
        out = cp.zeros(2, dtype=dtype)
        assert solver.setup(csr) is solver
        ptrs = {"csr": tuple(a.data.ptr for a in (csr.indptr, csr.indices, csr.data)),
                "rhs": rhs.data.ptr, "solution": out.data.ptr}
        for scale in (1, 2, 3):
            rhs[:] = cp.asarray([6 * scale, 7 * scale], dtype=dtype)
            out.fill(0)
            assert solver.solve(rhs, out=out) is out
            cp.testing.assert_allclose(out, [scale, 2 * scale], rtol=2e-5)
            assert solver.attached_ptrs == ptrs
            assert solver.status == "success"
        assert solver.iterations_number > 0
    # Destroying the wrapper does not free caller-owned data.
    cp.testing.assert_allclose(out, [3, 6], rtol=2e-5)
    cp.testing.assert_allclose(csr.data, [4, 1, 1, 3])


def test_rebinding_and_coefficient_refresh():
    cfg = config()
    cfg["solver"]["preconditioner"] = {"solver": "BLOCK_JACOBI", "max_iters": 1}
    with pyamgx.ReusableSolver(cfg) as solver:
        csr = matrix()
        rhs = cp.asarray([6., 7.])
        out = cp.zeros(2)
        solver.setup(csr).solve(rhs, out=out)
        cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)
        csr.data *= 2
        rhs *= 2
        out.fill(0)
        solver.setup(csr).solve(rhs, out=out)
        cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)
        rhs2, out2 = rhs * 3, cp.zeros(2)
        solver.solve(rhs2, out=out2)
        cp.testing.assert_allclose(out2, [3, 6], rtol=1e-6)
        assert solver.attached_ptrs["rhs"] == rhs2.data.ptr
        assert solver.attached_ptrs["solution"] == out2.data.ptr
        # A replacement buffer on the same CSR object is rebound at setup.
        old_data = weakref.ref(csr.data)
        csr.data = csr.data * 2
        out2.fill(0)
        solver.setup(csr).solve(rhs2 * 2, out=out2)
        gc.collect()
        assert old_data() is None
        cp.testing.assert_allclose(out2, [3, 6], rtol=1e-6)
        # A new CSR object and dimensions release the previous borrow.
        old_csr = weakref.ref(csr)
        del csr
        csr = sparse.eye(3, format="csr", dtype=cp.float64)
        solver.setup(csr).solve(cp.asarray([2., 3., 4.]), out=cp.zeros(3))
        assert old_csr() is None


def test_destroy_is_idempotent_and_releases_owners():
    solver = pyamgx.ReusableSolver(config())
    csr, rhs, out = matrix(), cp.asarray([6., 7.]), cp.zeros(2)
    refs = [weakref.ref(a) for a in (csr, csr.data, rhs, out)]
    solver.setup(csr).solve(rhs, out=out)
    del csr, rhs, out
    gc.collect()
    assert all(ref() is not None for ref in refs)
    solver.destroy()
    solver.destroy()
    gc.collect()
    assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="destroyed"):
        solver.setup(matrix())
    with pytest.raises(RuntimeError, match="destroyed"):
        solver.solve(cp.ones(2), out=cp.zeros(2))


def test_context_exception_and_gc_cleanup():
    with pytest.raises(ValueError, match="intentional"):
        with pyamgx.ReusableSolver(config()) as solver:
            solver.setup(matrix())
            raise ValueError("intentional")
    solver.destroy()
    solver = pyamgx.ReusableSolver(config())
    csr = matrix()
    ref = weakref.ref(csr.data)
    solver.setup(csr).solve(cp.asarray([6., 7.]), out=cp.zeros(2))
    del csr, solver
    gc.collect()
    assert ref() is None
    with pyamgx.ReusableSolver(config()) as fresh:
        fresh.setup(matrix())


def test_failed_setup_disables_solve_and_recovers():
    with pyamgx.ReusableSolver(config()) as solver:
        good = matrix()
        rhs, out = cp.asarray([6., 7.]), cp.zeros(2)
        with pytest.raises(RuntimeError, match="setup"):
            solver.solve(rhs, out=out)
        solver.setup(good).solve(rhs, out=out)
        bad = sparse.csr_matrix((cp.ones(2), cp.asarray([1, 0], dtype=cp.int32),
                                 cp.arange(3, dtype=cp.int32)), shape=(2, 2))
        with pytest.raises(pyamgx.AMGXError):
            solver.setup(bad)
        with pytest.raises(RuntimeError, match="setup"):
            solver.solve(rhs, out=out)
        out.fill(0)
        solver.setup(good).solve(rhs, out=out)
        cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)


def test_bad_vectors_and_mutated_storage_recover():
    with pyamgx.ReusableSolver(config()) as solver:
        csr, rhs, out = matrix(), cp.asarray([6., 7.]), cp.zeros(2)
        solver.setup(csr)
        with pytest.raises(ValueError, match="dtype"):
            solver.solve(rhs, out=cp.zeros(2, dtype=cp.float32))
        with pytest.raises(ValueError, match="dimension"):
            solver.solve(rhs, out=cp.zeros(3))
        with pytest.raises(pyamgx.AMGXError):
            solver.solve(rhs, out=rhs)
        solver.solve(rhs, out=out)
        rhs.shape = (1, 2)
        with pytest.raises(ValueError, match="one-dimensional"):
            solver.solve(rhs, out=out)
        rhs.shape = (2,)
        original = csr.data
        csr.data = original.copy()
        with pytest.raises(ValueError, match="storage changed"):
            solver.solve(rhs, out=out)
        out.fill(0)
        solver.setup(csr).solve(rhs, out=out)
        cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)


def test_destroying_one_solver_preserves_another():
    first = pyamgx.ReusableSolver(config())
    try:
        first.setup(matrix()).solve(cp.asarray([6., 7.]), out=cp.zeros(2))
        other_config = config()
        other_config["solver"]["max_iters"] = 40
        with pyamgx.ReusableSolver(other_config) as second:
            second.setup(matrix())
            first.destroy()
            out = cp.zeros(2)
            second.solve(cp.asarray([6., 7.]), out=out)
            cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)
            different_resources = config()
            different_resources["determinism_flag"] = 1
            with pytest.raises(ValueError, match="top-level"):
                pyamgx.ReusableSolver(different_resources)
            out.fill(0)
            second.solve(cp.asarray([12., 14.]), out=out)
            cp.testing.assert_allclose(out, [2, 4], rtol=1e-6)
    finally:
        first.destroy()


def test_constructor_failure_does_not_release_live_resources():
    with pyamgx.ReusableSolver(config()) as healthy:
        bad = config()
        bad["solver"]["solver"] = "NO_SUCH_SOLVER"
        with pytest.raises(pyamgx.AMGXError):
            pyamgx.ReusableSolver(bad)
        out = cp.zeros(2)
        healthy.setup(matrix()).solve(cp.asarray([6., 7.]), out=out)
        cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)


def test_native_setup_failure_releases_resources():
    bad = config()
    bad["solver"]["preconditioner"] = {
        "solver": "MULTICOLOR_DILU", "reorder_cols_by_color": 1,
    }
    with pyamgx.ReusableSolver(bad) as solver:
        with pytest.raises(pyamgx.AMGXError):
            solver.setup(matrix())
        with pytest.raises(RuntimeError, match="setup"):
            solver.solve(cp.ones(2), out=cp.zeros(2))
    with pyamgx.ReusableSolver(config()) as healthy:
        out = cp.zeros(2)
        healthy.setup(matrix()).solve(cp.asarray([6., 7.]), out=out)
        cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)


def test_explicit_stream_for_v2_producers():
    class V2Array:
        def __init__(self, array):
            self.array = array

        @property
        def __cuda_array_interface__(self):
            desc = dict(self.array.__cuda_array_interface__)
            desc["version"] = 2
            desc.pop("stream", None)
            return desc

    stream = cp.cuda.Stream(non_blocking=True)
    with stream, pyamgx.ReusableSolver(config()) as solver:
        rhs, out = V2Array(cp.asarray([6., 7.])), V2Array(cp.zeros(2))
        solver.setup(matrix(), stream=stream.ptr)
        with pytest.raises(ValueError, match="explicit producer stream"):
            solver.solve(rhs, out=out)
        assert solver.solve(rhs, out=out, stream=stream.ptr) is out
        cp.testing.assert_allclose(out.array, [1, 2], rtol=1e-6)
        rhs.array *= 2
        solver.solve(rhs, out=out, stream=stream.ptr)
        cp.testing.assert_allclose(out.array, [2, 4], rtol=1e-6)


def test_reentrant_cleanup_is_rejected():
    with pyamgx.ReusableSolver(config()) as solver:
        solver.setup(matrix())
        rhs = cp.asarray([6., 7.])

        class CheckedArray:
            @property
            def __cuda_array_interface__(self):
                with pytest.raises(RuntimeError, match="in use"):
                    solver.destroy()
                return rhs.__cuda_array_interface__

        out = cp.zeros(2)
        solver.solve(CheckedArray(), out=out)
        cp.testing.assert_allclose(out, [1, 2], rtol=1e-6)


@pytest.mark.parametrize("change", [
    {"exception_handling": 1}, {"structure_reuse_levels": -1},
])
def test_unsafe_config_rejected(change):
    cfg = config()
    cfg.update(change)
    with pytest.raises(ValueError):
        pyamgx.ReusableSolver(cfg)


def test_mixed_mode_rejected():
    with pytest.raises(ValueError, match="dDDI"):
        pyamgx.ReusableSolver(config(), "dDFI")
