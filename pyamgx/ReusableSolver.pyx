from copy import deepcopy as _reusable_deepcopy


cdef class _ReusableResources:
    cdef Config config
    cdef Resources resources
    cdef object root_config
    cdef int users
    cdef bint busy


# AMGX resource destruction releases global device pools and BLAS handles.
# Keep one resource handle alive until the last managed solver is destroyed.
cdef _ReusableResources _reusable_resources = None


cdef _ReusableResources _acquire_reusable_resources(dict config):
    global _reusable_resources
    cdef _ReusableResources state = _reusable_resources
    root = {key: value for key, value in config.items() if key != "solver"}
    if state is not None:
        if state.busy:
            raise RuntimeError("another reusable solver is in use")
        if state.root_config != root:
            raise ValueError("live reusable solvers must use the same top-level configuration; "
                             "put solver-specific options inside 'solver'")
    else:
        state = _ReusableResources()
        state.root_config = root
        state.config = Config().create_from_dict(config)
        try:
            state.resources = Resources().create_simple(state.config)
        except BaseException:
            state.config.destroy()
            raise
        _reusable_resources = state
    state.users += 1
    return state


def _check_reusable_config(config):
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    config = _reusable_deepcopy(config)
    config.setdefault("config_version", 2)
    config.setdefault("exception_handling", 0)
    if config["config_version"] != 2 or config["exception_handling"] != 0:
        raise ValueError("ReusableSolver requires config_version=2 and exception_handling=0")
    if not isinstance(config.get("solver"), dict):
        raise ValueError("config must contain a 'solver' dictionary")
    pending = [config]
    while pending:
        scope = pending.pop()
        if scope.get("structure_reuse_levels") == -1:
            raise ValueError("structure_reuse_levels=-1 skips coefficient refresh; use 0")
        pending.extend(value for value in scope.values() if isinstance(value, dict))
    return config


@cython.no_gc_clear
cdef class ReusableSolver:
    """Own reusable AMGX handles, borrowing CSR/RHS/output CUDA buffers.

    Call pyamgx.initialize() first and finalize() after all solvers are destroyed.
    setup(csr) builds or refreshes the preconditioner; solve(rhs, out=solution)
    writes directly into the supplied output and returns that same object.
    Use destroy() or a with block for deterministic cleanup.
    """
    cdef _ReusableResources _lease
    cdef Config _config
    cdef Matrix _matrix
    cdef Vector _rhs
    cdef Vector _solution
    cdef Solver _solver
    cdef object _mode
    cdef bint _closed
    cdef bint _constructed
    cdef bint _ready
    cdef bint _solved

    def __cinit__(self):
        self._closed = True

    def __init__(self, config, mode="dDDI"):
        if self._constructed:
            raise RuntimeError("ReusableSolver cannot be initialized twice")
        self._constructed = True
        if mode not in ("dDDI", "dFFI"):
            raise ValueError("ReusableSolver supports dDDI (float64) and dFFI (float32)")
        config = _check_reusable_config(config)
        self._mode = mode
        try:
            self._lease = _acquire_reusable_resources(config)
            self._config = Config().create_from_dict(config)
            self._matrix = Matrix().create(self._lease.resources, mode)
            self._rhs = Vector().create(self._lease.resources, mode)
            self._solution = Vector().create(self._lease.resources, mode)
            self._solver = Solver().create(self._lease.resources, self._config, mode)
        except BaseException:
            self._destroy_owned()
            raise
        self._closed = False

    cdef void _check_available(self) except *:
        if self._closed:
            raise RuntimeError("reusable solver has been destroyed")
        if self._lease.busy:
            raise RuntimeError("a reusable solver is in use")

    def setup(self, csr, *, stream=None):
        """Borrow CSR and build/refresh setup; return self.

        In-place coefficient updates reuse the matrix and solver handles.
        Replacing the CSR object or its storage recreates the native solver
        before releasing the old matrix borrow. No CSR data are copied.
        """
        self._check_available()
        self._ready = self._solved = False
        self._lease.busy = True
        try:
            shape, arrays, descriptors = _attached_csr_descriptor(
                csr, self._matrix._dtype, stream)
            same_storage = (self._matrix._owner is csr
                and shape == (self._matrix.shape[0], self._matrix.shape[1])
                and all(a is b for a, b in zip(arrays, self._matrix._arrays))
                and all(a[:3] == b[:3] for a, b in zip(descriptors, self._matrix._descriptors)))
            if not same_storage:
                if self._matrix.is_attached:
                    if self._solver is not None:
                        self._solver.destroy()
                        self._solver = None
                    self._matrix.detach()
                self._matrix.attach_CSR(csr, stream=stream)
            else:
                self._matrix._stream_override = stream
            if self._solver is None:
                self._solver = Solver().create(self._lease.resources, self._config, self._mode)
            self._solver.setup(self._matrix)
            self._ready = True
        finally:
            self._lease.busy = False
        return self

    def solve(self, rhs, *, out, stream=None):
        """Solve into out and return out, reusing setup and attached handles.

        Identical array objects take the fast path: no detach/attach or setup.
        Changed vector objects are rebound without copying. Buffer descriptors
        and producer streams are still checked by the underlying solve.
        out supplies the initial guess; fill it with zeros for a fresh solve.
        """
        self._check_available()
        if not self._ready:
            raise RuntimeError("solve() requires a successful setup()")
        self._solved = False
        self._lease.busy = True
        try:
            if self._rhs._owner is not rhs:
                if self._rhs.is_attached:
                    self._rhs.detach()
                self._rhs.attach(rhs, stream=stream)
            if self._solution._owner is not out:
                if self._solution.is_attached:
                    self._solution.detach()
                self._solution.attach(out, stream=stream)
            # A stream override applies to all five buffers for this call.
            self._matrix._stream_override = stream
            self._rhs._stream_override = stream
            self._solution._stream_override = stream
            self._solver.solve(self._rhs, self._solution)
            self._solved = True
        finally:
            self._lease.busy = False
        return out

    @property
    def attached_ptrs(self):
        """Native pointers for diagnostics; querying is outside the hot path."""
        self._check_available()
        return {
            "csr": self._matrix.attached_ptrs if self._matrix.is_attached else None,
            "rhs": self._rhs.attached_ptr if self._rhs.is_attached else None,
            "solution": self._solution.attached_ptr if self._solution.is_attached else None,
        }

    @property
    def status(self):
        self._check_available()
        if not self._solved:
            raise RuntimeError("no completed solve")
        return self._solver.status

    @property
    def iterations_number(self):
        self._check_available()
        if not self._solved:
            raise RuntimeError("no completed solve")
        return self._solver.iterations_number

    cdef void _destroy_owned(self) except *:
        global _reusable_resources
        self._ready = self._solved = False
        if self._solver is not None:
            self._solver.destroy()
            self._solver = None
        if self._solution is not None:
            self._solution.destroy()
            self._solution = None
        if self._rhs is not None:
            self._rhs.destroy()
            self._rhs = None
        if self._matrix is not None:
            self._matrix.destroy()
            self._matrix = None
        if self._config is not None:
            self._config.destroy()
            self._config = None
        if self._lease is not None:
            if self._lease.users == 1:
                self._lease.resources.destroy()
                self._lease.config.destroy()
                _reusable_resources = None
            self._lease.users -= 1
            self._lease = None
        self._closed = True

    def destroy(self):
        """Release owned handles and buffer references; safe to call repeatedly."""
        if self._closed:
            return
        self._check_available()
        self._destroy_owned()

    def __enter__(self):
        self._check_available()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()
        return False

    def __dealloc__(self):
        # Explicit destroy()/context management is required before finalize().
        # Also release handles if an initialized application drops the object.
        try:
            self._destroy_owned()
        except BaseException:
            pass
