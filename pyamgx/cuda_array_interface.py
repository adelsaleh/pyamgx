"""Validation shared by Vector.attach and its CPU-only protocol tests.

No CuPy dependency: the native API validates CUDA pointer kind and device.
"""

import operator
import numpy as np


def _attached_descriptor(array, dtype, stream=None):
    """Return (pointer, element_count, byte_count, producer_stream).

    Read the protocol property once. CAI v2 requires an explicit producer
    stream; CAI v3's None means producer work is already complete.
    """
    try:
        desc = array.__cuda_array_interface__
    except AttributeError:
        raise TypeError("attach requires a CUDA array-interface object") from None
    if not isinstance(desc, dict):
        raise TypeError("CUDA array interface must be a dictionary")
    version = desc.get("version")
    if type(version) is not int or version not in (2, 3):
        raise ValueError("attach supports CUDA array interface versions 2 and 3")
    if desc.get("mask") is not None:
        raise ValueError("masked CUDA arrays cannot be attached")
    try:
        actual_dtype = np.dtype(desc["typestr"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid CUDA array typestr") from None
    if actual_dtype.fields or not actual_dtype.isnative or actual_dtype != np.dtype(dtype):
        raise ValueError("attached dtype must match the vector's native real precision")
    descr = desc.get("descr")
    if descr is not None and descr != [("", desc["typestr"])]:
        raise ValueError("structured CUDA descriptors cannot be attached")
    shape = desc.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 1:
        raise ValueError("attached vectors must be one-dimensional")
    try:
        if isinstance(shape[0], (bool, np.bool_)):
            raise TypeError
        n = operator.index(shape[0])
    except TypeError:
        raise ValueError("invalid attached vector shape") from None
    if not 0 <= n <= np.iinfo(np.int32).max:
        raise ValueError("attached vector length must fit a nonnegative AMGX int")
    strides = desc.get("strides")
    if strides is not None:
        if not isinstance(strides, tuple) or len(strides) != 1:
            raise ValueError("invalid CUDA array strides")
        try:
            stride = operator.index(strides[0])
        except TypeError:
            raise ValueError("invalid CUDA array stride") from None
        if stride < 0 or (n > 1 and stride != actual_dtype.itemsize):
            raise ValueError("attached vectors must be contiguous without negative strides")
    data = desc.get("data")
    if not isinstance(data, tuple) or len(data) != 2 or type(data[1]) is not bool:
        raise ValueError("CUDA data must be a (pointer, read_only) tuple")
    if data[1]:
        raise ValueError("attached vectors must be writable")
    try:
        if isinstance(data[0], (bool, np.bool_)):
            raise TypeError
        ptr = operator.index(data[0])
    except TypeError:
        raise ValueError("invalid CUDA data pointer") from None
    max_pointer = np.iinfo(np.uintp).max
    byte_count = n * actual_dtype.itemsize
    if ptr < 0 or ptr > max_pointer or byte_count > max_pointer - ptr or (n and ptr == 0):
        raise ValueError("null or out-of-range CUDA data pointer")
    if ptr % actual_dtype.alignment:
        raise ValueError("misaligned CUDA data pointer")
    if stream is None:
        if version == 2:
            raise ValueError("CUDA array interface v2 requires an explicit producer stream")
        stream = desc.get("stream")
    if stream is None:
        stream = 0  # native API: the CAI producer promises completed work
    else:
        try:
            if isinstance(stream, (bool, np.bool_)):
                raise TypeError
            stream = operator.index(stream)
        except TypeError:
            raise ValueError("producer stream must be an integer CUDA stream handle") from None
        if not 1 <= stream <= max_pointer:
            raise ValueError("CUDA stream 0 is invalid; use 1 (legacy), 2 (PTDS), or a handle")
    return ptr, n, byte_count, stream


def _attached_csr_descriptor(csr, dtype, stream=None):
    """Describe three existing CSR buffers; never normalize or copy them.

    Structural contents are validated on the device by AMGX. Strong references
    to each array must be retained independently of the mutable CSR container.
    """
    if getattr(csr, "format", None) != "csr":
        raise TypeError("attach_CSR requires CSR format")
    shape = getattr(csr, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("CSR shape must be a pair")
    dims = []
    for dim in shape:
        if isinstance(dim, (bool, np.bool_)):
            raise ValueError("invalid CSR dimension")
        try:
            dims.append(operator.index(dim))
        except TypeError:
            raise ValueError("invalid CSR dimension") from None
    n, m = dims
    if n != m or not 0 < n < np.iinfo(np.int32).max:
        raise ValueError("attached CSR must be nonempty, square, and fit AMGX int32 dimensions")
    arrays = (csr.indptr, csr.indices, csr.data)
    descriptors = tuple(_attached_descriptor(a, t, stream)
                        for a, t in zip(arrays, (np.int32, np.int32, dtype)))
    rows, cols, values = descriptors
    if rows[1] != n + 1 or cols[1] != values[1]:
        raise ValueError("CSR buffer lengths do not match shape/nnz")
    if values[1] < n:
        raise ValueError("attached CSR requires an explicit diagonal in every row")
    for i, a in enumerate(descriptors):
        for b in descriptors[i + 1:]:
            if a[0] < b[0] + b[2] and b[0] < a[0] + a[2]:
                raise ValueError("CSR buffers must not overlap")
    return (n, m), arrays, descriptors
