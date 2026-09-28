cdef extern from "amgx_c.h":
    ctypedef enum  AMGX_SOLVE_STATUS:
        AMGX_SOLVE_SUCCESS = 0
        AMGX_SOLVE_FAILED = 1
        AMGX_SOLVE_DIVERGED = 2
        AMGX_SOLVE_NOT_CONVERGED = 3

@cython.no_gc_clear
cdef class Solver:
    """
    Solver: Class for creating and handling AMGX Solver objects.
    """
    cdef AMGX_solver_handle slv
    cdef Matrix A
    cdef list _matrices
    cdef object _resources
    cdef object _config
    cdef bint _busy
    cdef bint _ready

    def __cinit__(self):
        self.slv = NULL
        self._matrices = []

    cdef void _release_matrices(self):
        cdef Matrix matrix
        for matrix in self._matrices:
            matrix._users -= 1
        self._matrices = []
        self.A = None

    def __dealloc__(self):
        if self.slv != NULL:
            AMGX_solver_destroy(self.slv)
            self.slv = NULL
        self._release_matrices()

    cdef void _check_available(self) except *:
        if self.slv == NULL:
            raise RuntimeError("solver is not created or has been destroyed")
        if self._busy:
            raise RuntimeError("solver is in use by an AMGX operation")

    def create(self, Resources rsrc, Config cfg, mode='dDDI'):
        """
        solver.create(Resources rsrc, Config cfg, mode='dDDI')

        Create the underlying AMGX Solver object

        Parameters
        ----------
        rsrc : Resources
        cfg : Config
        mode : str, optional
            String representing data modes to use.
        """
        if self.slv != NULL:
            raise RuntimeError("solver is already created")
        check_error(AMGX_solver_create(
            &self.slv, rsrc.rsrc, asMode(mode),
            cfg.cfg))
        self._resources, self._config = rsrc, cfg
        return self

    def setup(self, Matrix A):
        """
        solver.setup(A)

        Invoke the set up phase.

        Parameters
        ----------
        A : Matrix
            The `Matrix` representing the coefficient matrix of the equation
            to be solved.
        """
        cdef AMGX_RC status
        self._check_available()
        A._begin_use()
        self._busy = True
        self._ready = False
        try:
            # Native setup may retain the matrix even when it reports an error.
            # Keep every attempted matrix until destruction, including failures
            # before native setup replaces its previous shared_ptr.
            if A not in self._matrices:
                self._matrices.append(A)
                A._users += 1
            with nogil:
                status = AMGX_solver_setup(self.slv, A.mtx)
            check_error(status)
            self.A = A
            self._ready = True
        finally:
            self._busy = False
            A._end_use()

    def solve(self, Vector b, Vector x, zero_initial_guess=False):
        """
        solver.solve(Vector rhs, Vector sol)

        Invoke the solve phase.

        Parameters
        ----------
        b: Vector
            The `Vector` representing the right-hand side of the equation
            to be solved.
        x: Vector
            The `Vector` representing the solution vector of the equation
            to be solved. If `zero_initial_guess` is unspecified,
            the values in `x` are used as the initial guess for iterative
            algorithms.
        zero_initial_guess : bool, optional
            If `True`, use an initial guess of zero for the solution,
            regardless of the values in `x`.
        """
        self._check_available()
        if self.A is None or not self._ready:
            raise RuntimeError, "solve() cannot be called before setup()"

        A_size, _ = self.A.get_size()
        b_size, _ = b.get_size()
        x_size, _ = x.get_size()

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError, "Matrix is not square: {} != {}".format(self.A.shape[0], self.A.shape[1])
        if A_size != b_size:
            raise ValueError, "Matrix - RHS dimension mismatch: {} != {}".format(
                A_size, b_size)
        if b_size != x_size:
            raise ValueError, "RHS - solution dimension mismatch: {} != {}".format(
                b_size, x_size)

        cdef AMGX_RC status
        self.A._begin_use()
        self._busy = True
        try:
            b._begin_use()
            try:
                x._begin_use()
                try:
                    if zero_initial_guess:
                        with nogil:
                            status = AMGX_solver_solve_with_0_initial_guess(self.slv, b.vec, x.vec)
                    else:
                        with nogil:
                            status = AMGX_solver_solve(self.slv, b.vec, x.vec)
                    check_error(status)
                finally:
                    x._end_use()
            finally:
                b._end_use()
        finally:
            self._busy = False
            self.A._end_use()

    @property
    def status(self):
        """
        solver.status

        The status from the last solve phase.
        """
        cdef AMGX_SOLVE_STATUS stat
        check_error(AMGX_solver_get_status(self.slv, &stat))
        if stat is AMGX_SOLVE_SUCCESS:
            return 'success'
        elif stat == AMGX_SOLVE_FAILED:
            return 'failed'
        elif stat == AMGX_SOLVE_DIVERGED:
            return 'diverged'
        elif stat == AMGX_SOLVE_NOT_CONVERGED:
            return 'not_converged'
        else:
            raise ValueError, 'Invalid solver status returned.'

    @property
    def iterations_number(self):
        """
        solver.iterations_number

        The number of iterations that were executed during the last
        solve phase.
        """
        cdef int niter
        check_error(AMGX_solver_get_iterations_number(self.slv, &niter))
        return niter

    def get_residual(self, int iteration=-1, int block_idx=0):
        """
        solver.get_residual(iteration, block_idx)

        Get the value of the residual for a given iteration from the
        last solve phase.

        Parameters
        ----------
        iteration : int, optional
            The iteration at which to inspect the residual. If
            not provided, return the residual after
            the last iteration.

        block_idx : int, optional
            The index of the entry in the block of the residual
            to retrieve.
        Returns
        -------
        res : float
            Value of residual at specified iteration.
        """
        cdef double res
        if iteration == - 1:
            iteration = self.iterations_number
        check_error(AMGX_solver_get_iteration_residual(self.slv,
            iteration, block_idx, &res))
        return res

    def destroy(self):
        """
        solver.destroy()

        Destroy the underlying AMGX Solver object.
        """
        self._check_available()
        check_error(AMGX_solver_destroy(self.slv))
        self.slv = NULL
        self._ready = False
        self._release_matrices()
        self._resources = self._config = None
