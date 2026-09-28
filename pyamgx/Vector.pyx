from libc.stdint cimport uintptr_t
cimport cython

@cython.no_gc_clear
cdef class Vector:
    """
    Vector: Class for creating and handling AMGX Vector objects.

    Examples
    --------

    Creating a vector, uploading values from a numpy array,
    and downloading values back to a numpy array:

    >>> import pyamgx, numpy as np
    >>> pyamgx.initialize()
    >>> cfg = pyamgx.Config().create("")
    >>> rsrc = pyamgx.Resources().create_simple(cfg)
    >>> v = pyamgx.Vector().create(rsrc)
    >>> v.upload(np.array([1., 2., 3.,], dtype=np.float64))
    >>> v.download()
    array([ 1.,  1.,  1.])
    >>> v.destroy()
    >>> rsrc.destroy()
    >>> cfg.destroy()
    >>> pyamgx.finalize()


    """
    cdef AMGX_vector_handle vec
    cdef object _owner
    cdef object _dtype
    cdef object _stream_override
    cdef object _mode
    cdef Resources _resources
    cdef uintptr_t _attached_ptr
    cdef size_t _attached_bytes
    cdef bint _busy

    def __cinit__(self):
        self.vec = NULL
        self._owner = None
        self._busy = False

    def __dealloc__(self):
        # A borrowed allocation must never outlive its Python owner reference.
        # The C destroy waits for AMGX completion before releasing the handle.
        if self.vec != NULL and self._owner is not None:
            AMGX_vector_destroy(self.vec)
            self.vec = NULL

    cdef void _check_available(self) except *:
        if self.vec == NULL:
            raise RuntimeError("vector is not created or has been destroyed")
        if self._busy:
            raise RuntimeError("vector is in use by an AMGX operation")

    cdef void _begin_use(self) except *:
        self._check_available()
        self._busy = True
        try:
            if self._owner is not None:
                ptr, n, byte_count, stream = _attached_descriptor(
                    self._owner, self._dtype, self._stream_override)
                if ptr != self._attached_ptr or byte_count != self._attached_bytes:
                    raise ValueError("attached array storage changed; detach and attach again")
                check_error(AMGX_vector_synchronize(self.vec, stream))
        except BaseException:
            self._busy = False
            raise

    cdef void _end_use(self):
        self._busy = False

    @property
    def is_attached(self):
        return self._owner is not None

    @property
    def attached_ptr(self):
        """Native borrowed pointer, for interoperability identity checks."""
        self._check_available()
        cdef void *ptr
        cdef size_t byte_count
        cdef int device
        check_error(AMGX_vector_get_attached_data(self.vec, &ptr, &byte_count, &device))
        return <uintptr_t>ptr

    def attach(self, data, *, stream=None):
        """Borrow a writable contiguous CUDA vector, without copying.

        Only scalar real device modes are supported. The vector must be empty.
        Keep the array's allocation stable and do not access it concurrently
        with AMGX. This object retains the producer until detach/destroy.
        CAI v3 supplies the producer stream; v2 requires an explicit integer
        stream. The descriptor is checked again before each solve, so later
        writes must be ordered on the producer's exported stream.
        """
        self._check_available()
        if self._mode not in ('dDDI', 'dDFI', 'dFFI'):
            raise ValueError("attach requires a real CUDA vector mode")
        if self._owner is not None:
            raise RuntimeError("vector is already attached; detach first")
        ptr, n, byte_count, producer_stream = _attached_descriptor(data, self._dtype, stream)
        check_error(AMGX_vector_attach(self.vec, n, <void *><uintptr_t>ptr,
                                      byte_count, producer_stream))
        self._owner = data
        self._stream_override = stream
        self._attached_ptr = ptr
        self._attached_bytes = byte_count
        return self

    def detach(self):
        """Release the borrow after completion, returning its owner without copying.

        The native vector becomes empty and can be attached or uploaded again.
        """
        self._check_available()
        if self._owner is None:
            raise RuntimeError("vector is not attached")
        check_error(AMGX_vector_detach(self.vec))
        owner = self._owner
        self._owner = None
        self._stream_override = None
        self._attached_ptr = 0
        self._attached_bytes = 0
        return owner

    def create(self, Resources rsrc, mode='dDDI'):
        """
        v.create(Resources rsrc, mode='dDDI')

        Create the underlying AMGX Vector object.

        Parameters
        ----------
        rsrc : Resources
        mode : str, optional
            String representing data modes to use.

        Returns
        -------
        self : Vector
        """
        if self.vec != NULL:
            raise RuntimeError("vector is already created")
        check_error(AMGX_vector_create(&self.vec, rsrc.rsrc, asMode(mode)))
        self._mode = mode
        self._dtype = {'D': np.dtype('float64'), 'F': np.dtype('float32')}.get(mode[1])
        self._resources = rsrc
        return self

    def upload(self, data, block_dim=1):
        """
        v.upload(data, block_dim=1)

        Copy data to the Vector from an array.

        Parameters
        ----------
        data : array_like, ndim=1
            Array to copy data from.

        block_dim : int, optional
            Number of values per block.

        Returns
        -------
        self : Vector
        """

        self._check_available()
        if self._dtype is None:
            raise ValueError("upload supports real vector precision only")
        block_dim = operator.index(block_dim)
        if block_dim < 1 or data.size % block_dim:
            raise ValueError("vector size must be divisible by a positive block dimension")
        n = data.size // block_dim

        cdef uintptr_t ptr = ptr_from_array_interface(data, self._dtype)
        self.upload_raw(ptr, n, block_dim)

        return self

    def upload_raw(self, uintptr_t ptr, int n, block_dim=1):
        """
        v.upload_raw(ptr, n, block_dim=1)

        Copy data to the Vector from an array, given a raw pointer
        to the array.

        Parameters
        ----------
        ptr : pointer
            An integer (or long integer, if required) that
            points to the array containing data
        n : int
            Size of the array
        block_dim : int, optional
            Number of values per block.
        """

        self._check_available()
        if self._owner is not None:
            raise RuntimeError("upload would copy into borrowed storage; detach first")
        check_error(AMGX_vector_upload(
            self.vec, n, block_dim,
            <void *> ptr))

        return self

    def download(self, data=None):
        """
        v.download(data)

        Copy data from the Vector to an array.

        Parameters
        ----------
        data : array, ndim=1
            Array to copy data to.
        """
        self._check_available()
        n, block_dim = self.get_size()
        size = n * block_dim
        if self._dtype is None:
            raise ValueError("download supports real vector precision only")
        if data is None:
            data = np.empty(size, dtype=self._dtype)
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(memoryview(data))
            except TypeError:
                raise TypeError("download requires a host buffer; attached CUDA solutions need no download") from None
        if (data.ndim != 1 or data.size < size or data.dtype != self._dtype
                or not data.flags.c_contiguous or not data.flags.writeable):
            raise ValueError("download requires a writable contiguous NumPy vector with sufficient capacity and matching dtype")
        if size:
            self.download_raw(data.ctypes.data)
        return data

    def download_raw(self, uintptr_t ptr):
        """
        v.download_raw(ptr)

        Copy data from the Vector to an array, given a raw pointer
        to the array.

        Parameters
        ----------
        ptr : pointer
            An integer (or long integer, if required) that
            points to the array containing data
        """
        self._begin_use()
        try:
            check_error(AMGX_vector_download(self.vec, <void *>ptr))
        finally:
            self._end_use()

    def set_zero(self, n=None, block_dim=None):
        """
        v.set_zero(n=None, block_dim=None)

        Allocate storage if needed and set all
        the values in the vector to zero.

        Parameters
        ----------
        n : int, optional
            Number of entries in the vector, in block units.
            If not provided or `None`, then the values *must*
            have been previously initialized using e.g.,
            `v.upload()`.

        block_dim : int, optional
            Number of values per block. If not provided
            or `None`, then the values *must* have been previously

            initialized using e.g., `v.upload()`.
        """
        n_, block_dim_ = self.get_size()
        if n_ == 0:
            if n is None or block_dim is None:
                raise ValueError("set_zero() requires arguments"
                                 "'n' and 'block_dim'"
                                 "for uninitialized vector")
        if n is None:
            n = n_
        if block_dim is None:
            block_dim = block_dim_

        self._begin_use()
        try:
            check_error(AMGX_vector_set_zero(self.vec, n, block_dim))
        finally:
            self._end_use()

    def get_size(self):
        """
        v.get_size()

        Get the size of the vector (in block units), and the
        block size.

        Returns
        -------
        n : int
            The size of the vector in block units.
        block_dim : int
            The block size.
        """
        self._check_available()
        cdef int n, block_dim
        check_error(AMGX_vector_get_size(
            self.vec,
            &n, &block_dim))
        return n, block_dim

    def destroy(self):
        """
        v.destroy()

        Destroy the underlying AMGX Vector object.
        """
        self._check_available()
        check_error(AMGX_vector_destroy(self.vec))
        self.vec = NULL
        self._owner = None
        self._resources = None
