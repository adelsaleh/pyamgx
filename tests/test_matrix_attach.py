"""Small CuPyX CSR regressions, requiring the rebuilt attachment ABI."""
import gc
import weakref

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from cupyx.scipy import sparse
import pyamgx
from test_vector_attach import queue_delay


@pytest.fixture
def handles():
    assert hasattr(pyamgx.Matrix, "attach_CSR"), "Build the CSR attachment extension"
    pyamgx.initialize()
    objects = []
    def own(obj):
        objects.append(obj)
        return obj
    def destroy(obj):
        obj.destroy()
        objects.remove(obj)
    cfg = own(pyamgx.Config().create_from_dict({
        "config_version": 2, "exception_handling": 0,
        "solver": {"solver": "FGMRES", "preconditioner": {"solver": "NOSOLVER"},
                   "max_iters": 30, "gmres_n_restart": 8, "tolerance": 1e-6,
                   "monitor_residual": 1, "convergence": "RELATIVE_INI_CORE"}}))
    resources = own(pyamgx.Resources().create_simple(cfg))
    try:
        yield own, destroy, cfg, resources
    finally:
        for obj in reversed(objects):
            obj.destroy()
        pyamgx.finalize()


def matrix(dtype):
    return sparse.csr_matrix(cp.asarray([[4, 1, 0], [1, 4, 1], [0, 1, 3]], dtype=dtype))


def pointers(a):
    return tuple(v.data.ptr for v in (a.indptr, a.indices, a.data))


@pytest.mark.parametrize("mode,dtype", [("dDDI", np.float64), ("dFFI", np.float32)])
@pytest.mark.parametrize("stream_kind", ["legacy", "ptds", "nonblocking"])
def test_borrowed_csr_solve_and_value_update(handles, mode, dtype, stream_kind):
    own, destroy, cfg, resources = handles
    stream = {"legacy": cp.cuda.Stream.null, "ptds": cp.cuda.Stream.ptds,
              "nonblocking": cp.cuda.Stream(non_blocking=True)}[stream_kind]
    with stream:
        a = matrix(dtype)
        pattern = (a.indptr.copy(), a.indices.copy())
        ptrs = pointers(a)
        expected = cp.asarray([1, 2, 3], dtype=dtype)
        rhs = a @ expected
        result = cp.zeros(3, dtype=dtype)
        queue_delay(stream)
        A = own(pyamgx.Matrix().create(resources, mode)).attach_CSR(a)
        b = own(pyamgx.Vector().create(resources, mode)).attach(rhs)
        x = own(pyamgx.Vector().create(resources, mode)).attach(result)
        solver = own(pyamgx.Solver().create(resources, cfg, mode))
        for scale in (1, 2):
            queue_delay(stream)
            a.data *= scale
            rhs[:] = a @ expected
            result.fill(0)
            solver.setup(A)
            solver.solve(b, x)
            assert A.attached_ptrs == ptrs == pointers(a)
            # A different stream consumes the completed solve immediately.
            with cp.cuda.Stream(non_blocking=True):
                cp.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)
                assert float(cp.linalg.norm(rhs - a @ result)) < 1e-4
            cp.testing.assert_array_equal(a.indptr, pattern[0])
            cp.testing.assert_array_equal(a.indices, pattern[1])
        with pytest.raises(RuntimeError, match="solvers"):
            A.detach()
        with pytest.raises(RuntimeError, match="solvers"):
            A.destroy()
        destroy(solver)
        assert A.detach() is a
        A.attach_CSR(a)
        assert A.attached_ptrs == ptrs


def test_owner_retention_and_changed_buffer(handles):
    own, destroy, cfg, resources = handles
    a = matrix(np.float64)
    original = a.data
    ref = weakref.ref(original)
    A = own(pyamgx.Matrix().create(resources)).attach_CSR(a)
    a.data = a.data.copy()
    del original
    gc.collect()
    assert ref() is not None
    solver = own(pyamgx.Solver().create(resources, cfg))
    with pytest.raises(ValueError, match="storage changed"):
        solver.setup(A)
    destroy(solver)
    A.detach()
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("invalid", ["row_end", "descending", "duplicate", "column", "diagonal"])
def test_native_rejects_invalid_structure_without_borrowing(handles, invalid):
    own, _, _, resources = handles
    a = matrix(np.float64)
    if invalid == "row_end": a.indptr[-1] -= 1
    if invalid == "descending": a.indptr[1] = -1
    if invalid == "duplicate": a.indices[1] = 0
    if invalid == "column": a.indices[0] = 3
    if invalid == "diagonal":
        # Valid, sorted, duplicate-free CSR whose three diagonal entries are
        # absent. This isolates the diagonal check from column-order rejection.
        a = sparse.csr_matrix((cp.ones(3, dtype=cp.float64),
                               cp.asarray([1, 2, 0], dtype=cp.int32),
                               cp.arange(4, dtype=cp.int32)), shape=(3, 3))
    A = own(pyamgx.Matrix().create(resources))
    with pytest.raises(pyamgx.AMGXError):
        A.attach_CSR(a)
    assert not A.is_attached
    a = matrix(np.float64)
    A.attach_CSR(a)
    assert A.attached_ptrs == pointers(a)


def test_copy_api_rejected_while_attached(handles):
    own, _, _, resources = handles
    a = matrix(np.float64)
    A = own(pyamgx.Matrix().create(resources)).attach_CSR(a)
    with pytest.raises(RuntimeError): A.upload_CSR(a)
    with pytest.raises(RuntimeError): A.replace_coefficients(a.data)
    assert A.attached_ptrs == pointers(a)


@pytest.mark.parametrize("preconditioner", [
    {"solver": "BLOCK_JACOBI"},
    {"solver": "AMG", "algorithm": "CLASSICAL", "max_levels": 5,
     "max_iters": 1, "cycle": "V", "coarse_solver": "DENSE_LU_SOLVER",
     "smoother": {"solver": "JACOBI_L1"}, "min_coarse_rows": 4},
])
@pytest.mark.parametrize("mode,dtype", [("dDDI", np.float64), ("dFFI", np.float32)])
def test_preconditioned_solve_preserves_fine_csr(handles, preconditioner, mode, dtype):
    own, _, _, resources = handles
    cfg = own(pyamgx.Config().create_from_dict({
        "config_version": 2, "exception_handling": 0,
        "solver": {"solver": "FGMRES", "preconditioner": preconditioner,
                   "max_iters": 100, "gmres_n_restart": 20, "tolerance": 1e-6,
                   "monitor_residual": 1, "convergence": "RELATIVE_INI_CORE"}}))
    a = sparse.diags([cp.full(63, -1, dtype=dtype), cp.full(64, 4, dtype=dtype),
                     cp.full(63, -1, dtype=dtype)], [-1, 0, 1], format="csr")
    a.sort_indices()  # producer preparation happens before attachment
    saved = tuple(v.copy() for v in (a.indptr, a.indices, a.data))
    A = own(pyamgx.Matrix().create(resources, mode)).attach_CSR(a)
    expected = cp.linspace(1, 2, 64, dtype=dtype)
    rhs, result = a @ expected, cp.zeros(64, dtype=dtype)
    b = own(pyamgx.Vector().create(resources, mode)).attach(rhs)
    x = own(pyamgx.Vector().create(resources, mode)).attach(result)
    solver = own(pyamgx.Solver().create(resources, cfg, mode))
    solver.setup(A)
    solver.solve(b, x)
    assert A.attached_ptrs == pointers(a)
    cp.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)
    for actual, original in zip((a.indptr, a.indices, a.data), saved):
        cp.testing.assert_array_equal(actual, original)


def test_mixed_mode_retains_existing_solver_limitation(handles):
    own, _, cfg, resources = handles
    a = matrix(np.float32)
    A = own(pyamgx.Matrix().create(resources, 'dDFI')).attach_CSR(a)
    b = own(pyamgx.Vector().create(resources, 'dDFI')).attach(cp.ones(3, dtype=cp.float64))
    x = own(pyamgx.Vector().create(resources, 'dDFI')).attach(cp.zeros(3, dtype=cp.float64))
    solver = own(pyamgx.Solver().create(resources, cfg, 'dDFI'))
    solver.setup(A)
    with pytest.raises(pyamgx.AMGXError, match="not implemented"):
        solver.solve(b, x)
    assert A.attached_ptrs == pointers(a)


def test_mutating_setup_is_rejected_and_owner_survives(handles):
    own, destroy, _, resources = handles
    cfg = own(pyamgx.Config().create_from_dict({
        "config_version": 2, "exception_handling": 0,
        "solver": {"solver": "FGMRES", "preconditioner": {"solver": "MULTICOLOR_DILU", "reorder_cols_by_color": 1}}}))
    a = matrix(np.float64)
    saved = tuple(v.copy() for v in (a.indptr, a.indices, a.data))
    A = own(pyamgx.Matrix().create(resources)).attach_CSR(a)
    solver = own(pyamgx.Solver().create(resources, cfg))
    with pytest.raises(pyamgx.AMGXError):
        solver.setup(A)
    assert A.attached_ptrs == pointers(a)
    for actual, original in zip((a.indptr, a.indices, a.data), saved):
        cp.testing.assert_array_equal(actual, original)
    with pytest.raises(RuntimeError, match="solvers"):
        A.detach()
    destroy(solver)
    assert A.detach() is a


def test_garbage_collection_releases_owner_after_native_matrix(handles):
    _, _, _, resources = handles
    a = matrix(np.float64)
    data_ref = weakref.ref(a.data)
    A = pyamgx.Matrix().create(resources).attach_CSR(a)
    del a
    gc.collect()
    assert data_ref() is not None
    del A
    gc.collect()
    assert data_ref() is None
