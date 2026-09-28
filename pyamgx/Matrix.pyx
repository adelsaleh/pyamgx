cimport cython

@cython.no_gc_clear
cdef class Matrix:
    """
    `Matrix` : Class for creating and handling AMGX Matrix objects.

    Examples
    --------

    Uploading the matrix ``[[1, 2], [3, 4]]`` using the `upload` method:

    >>> import pyamgx, numpy as np
    >>> pyamgx.initialize()
    >>> cfg = pyamgx.Config().create("")
    >>> rsrc = pyamgx.Resources().create_simple(cfg)
    >>> M = pyamgx.Matrix().create(rsrc)
    >>> M.upload(
    ...     row_ptrs=np.array([0, 2, 4], dtype=np.int32),
    ...     col_indices=np.array([0, 1, 0, 1], dtype=np.int32),
    ...     data=np.array([1., 2., 3., 4.], dtype=np.float64))
    >>> M.destroy()
    >>> rsrc.destroy()
    >>> cfg.destroy()
    >>> pyamgx.finalize()

    """
    cdef AMGX_matrix_handle mtx
    cdef int shape[2]
    cdef object _dtype
    cdef object _owner
    cdef object _arrays
    cdef object _descriptors
    cdef object _stream_override
    cdef object _mode
    cdef Resources _resources
    cdef bint _busy
    cdef int _users

    def __cinit__(self):
        self.mtx = NULL
        self._owner = None
        self._busy = False
        self._users = 0

    def __dealloc__(self):
        if self.mtx != NULL:
            AMGX_matrix_destroy(self.mtx)
            self.mtx = NULL

    cdef void _check_available(self) except *:
        if self.mtx == NULL:
            raise RuntimeError("matrix is not created or has been destroyed")
        if self._busy:
            raise RuntimeError("matrix is in use by an AMGX operation")

    cdef void _begin_use(self) except *:
        self._check_available()
        if self._owner is not None:
            shape, arrays, descriptors = _attached_csr_descriptor(
                self._owner, self._dtype, self._stream_override)
            if shape != (self.shape[0], self.shape[1]) or any(
                    a is not b for a, b in zip(arrays, self._arrays)) or any(
                    a[:3] != b[:3] for a, b in zip(descriptors, self._descriptors)):
                raise ValueError("attached CSR storage changed; destroy its solvers and detach first")
            check_error(AMGX_matrix_synchronize(self.mtx,
                descriptors[0][3], descriptors[1][3], descriptors[2][3]))
        self._busy = True

    cdef void _end_use(self):
        self._busy = False

    @property
    def is_attached(self):
        return self._owner is not None

    @property
    def attached_ptrs(self):
        """Native (indptr, indices, data) addresses, for identity checks."""
        self._check_available()
        cdef void *rows
        cdef void *cols
        cdef void *data
        check_error(AMGX_matrix_get_attached_data(self.mtx, &rows, &cols, &data))
        return <uintptr_t>rows, <uintptr_t>cols, <uintptr_t>data

    def attach_CSR(self, csr, *, stream=None):
        """Borrow CuPyX CSR indptr/indices/data without copying.

        Requires canonical square CSR, int32 indices, matching real precision,
        and an explicit diagonal in each row. Structure is immutable while
        attached. After changing values in place, call solver.setup again.
        Destroy associated solvers before detaching or destroying the matrix.
        """
        self._check_available()
        if self._mode not in ('dDDI', 'dDFI', 'dFFI'):
            raise ValueError("attach_CSR requires a real CUDA matrix mode")
        if self._owner is not None or self._users:
            raise RuntimeError("matrix is attached or retained by a solver")
        shape, arrays, descriptors = _attached_csr_descriptor(csr, self._dtype, stream)
        rows, cols, data = descriptors
        check_error(AMGX_matrix_attach_csr(self.mtx, shape[0], data[1],
            <int *><uintptr_t>rows[0], <int *><uintptr_t>cols[0], <void *><uintptr_t>data[0],
            rows[2], cols[2], data[2], rows[3], cols[3], data[3]))
        self.shape = shape[0], shape[1]
        self._owner, self._arrays, self._descriptors = csr, arrays, descriptors
        self._stream_override = stream
        return self

    def detach(self):
        """Release all three borrowed buffers and return the CSR owner."""
        self._check_available()
        if self._owner is None:
            raise RuntimeError("matrix is not attached")
        if self._users:
            raise RuntimeError("destroy associated solvers before detaching the matrix")
        check_error(AMGX_matrix_detach(self.mtx))
        owner = self._owner
        self._owner = self._arrays = self._descriptors = self._stream_override = None
        self.shape = 0, 0
        return owner

    def create(self, Resources rsrc, mode='dDDI'):
        """
        M.create(Resources rsrc, mode='dDDI')

        Create the underlying AMGX Matrix object.

        Parameters
        ----------
        rsrc : Resources

        mode : str, optional
            String representing data modes to use.

        Returns
        -------
        self : Matrix
        """
        if self.mtx != NULL:
            raise RuntimeError("matrix is already created")
        check_error(AMGX_matrix_create(&self.mtx, rsrc.rsrc, asMode(mode)))
        self._resources = rsrc
        self._mode = mode
        self._dtype = {'D': np.dtype('float64'), 'F': np.dtype('float32')}.get(mode[2])
        return self

    def upload(self, row_ptrs, col_indices, data, block_dims=[1, 1], shape=None):
        """
        M.upload(row_ptrs, col_indices, data, block_dims=[1, 1])

        Copy data from arrays describing the sparse matrix to
        the Matrix object.

        Parameters
        ----------
        row_ptrs : array_like
            Array of row pointers. For a description of the arrays
            `row_ptrs`, `col_indices` and `data`,
            see `here <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_.
        col_indices : array_like
            Array of column indices.
        data : array_like
            Array of matrix data.
        block_dims : tuple_like, optional
            Dimensions of block in x- and y- directions. Currently
            only square blocks are supported, so block_dims[0] should be
            equal to block_dims[1].

        Returns
        -------
        self : Matrix
        """
        cdef int block_dimx, block_dimy
        cdef int nrows, ncols

        self._check_available()
        if self._owner is not None:
            raise RuntimeError("cannot upload into attached CSR; detach first")

        block_dimx = block_dims[0]
        block_dimy = block_dims[1]

        nnz = len(data)
        if shape is None:
            nrows = len(row_ptrs) - 1
            ncols = col_indices.max() + 1
        else:
            nrows = shape[0]
            ncols = shape[1]
        self.shape = nrows, ncols

        cdef uintptr_t row_ptrs_ptr = ptr_from_array_interface(
            row_ptrs, "int32"
        )
        cdef uintptr_t col_indices_ptr = ptr_from_array_interface(
            col_indices, "int32"
        )
        if self._dtype is None:
            raise ValueError("upload supports real matrix precision only")
        cdef uintptr_t data_ptr = ptr_from_array_interface(data, self._dtype)

        check_error(AMGX_matrix_upload_all(
            self.mtx,
            nrows, nnz, block_dimx, block_dimy,
            <const int*> row_ptrs_ptr, <const int*> col_indices_ptr,
            <void*> data_ptr, NULL)
        )

        return self

    def upload_CSR(self, csr):
        """
        M.upload_CSR(csr)

        Copy data from a :class:`scipy.sparse.csr_matrix` or CuPy sparse matrix
        to the Matrix object.

        Parameters
        ----------
        csr : scipy.sparse.csr_matrix

        Returns
        -------
        self : Matrix
        """
        nrows = csr.shape[0]
        ncols = csr.shape[1]

        row_ptrs = csr.indptr
        col_indices = csr.indices
        data = csr.data

        if len(col_indices) == 0:
            # assume matrix of zeros
            col_indices = col_indices.__class__((1,), dtype=np.int32)
            col_indices.fill(ncols-1)
            data = data.__class__((1,), dtype=self._dtype)
            data.fill(0)

        self.upload(row_ptrs, col_indices, data, shape=[nrows, ncols])
        return self

    def get_size(self):
        """
        M.get_size()

        Get the matrix size (in block units), and the block dimensions.

        Returns
        -------

        n : int
            The matrix size (number of rows/columns) in block units.
        block_dims : tuple
            A tuple (`bx`, `by`) representing the size of the
            blocks in the x- and y- dimensions.
        """
        cdef int n, bx, by
        self._check_available()
        check_error(AMGX_matrix_get_size(
            self.mtx,
            &n, &bx, &by))
        return n, (bx, by)

    def get_nnz(self):
        """
        M.get_nnz()

        Get the number of non-zero entries of the Matrix.

        Returns
        -------
        nnz : int
        """
        cdef int nnz
        self._check_available()
        check_error(AMGX_matrix_get_nnz(
            self.mtx,
            &nnz))
        return nnz

    def replace_coefficients(self, data):
        """
        M.replace_coefficients(data)
        Replace matrix coefficients without changing the nonzero structure.

        Parameters
        ----------
        data : array_like
            Array of matrix data.
        """
        cdef int n, nnz
        self._check_available()
        if self._owner is not None:
            raise RuntimeError("update attached CSR data in place, then repeat solver.setup")
        if self._dtype is None:
            raise ValueError("replace_coefficients supports real matrix precision only")
        cdef uintptr_t data_ptr = ptr_from_array_interface(data, check_for_dtype=self._dtype)
        
        size, (bx, by) = self.get_size()
        n = self.get_size()[0]
        nnz = self.get_nnz()
        check_error(AMGX_matrix_replace_coefficients(
            self.mtx, n, nnz, <void *> data_ptr, NULL))

        
    def destroy(self):
        """
        M.destroy()

        Destroy the underlying AMGX Matrix object.
        """
        self._check_available()
        if self._users:
            raise RuntimeError("destroy associated solvers before destroying the matrix")
        check_error(AMGX_matrix_destroy(self.mtx))
        self.mtx = NULL
        self._owner = self._arrays = self._descriptors = self._stream_override = None
        self._resources = None
