from cupy import cuda

def _as_batch_mat(x):
    return x.reshape(len(x), x.shape[1], -1)

def _mat_ptrs(a):
    if len(a) == 1:
        return cupy.full((1,), a.data.ptr, dtype=np.uintp)
    else:
        stride = a.strides[0]
        ptr = a.data.ptr
        out = cupy.arange(ptr, ptr + stride * len(a), stride, dtype=np.uintp)
        return out


def _get_ld(a):
    strides = a.strides[-2:]
    trans = np.argmin(strides)
    return trans, int(max(a.shape[trans - 2], max(strides) // a.itemsize))


def inv_gpu(b):
    # We do a batched LU decomposition on the GPU to compute the inverse
    # Change the shape of the array to be size=1 minibatch if necessary
    # Also copy the matrix as the elments will be modified in-place
    a = _as_batch_mat(b).copy()
    n = a.shape[1]
    n_matrices = len(a)
    # Pivot array
    p = cupy.empty((n, n_matrices), dtype=np.int32)
    # Output array
    c = cupy.empty_like(a)
    # These arrays hold information on the execution success
    # or if the matrix was singular
    info = cupy.empty(n_matrices, dtype=np.int32)
    ap = _mat_ptrs(a)
    cp = _mat_ptrs(c)
    _, lda = _get_ld(a)
    _, ldc = _get_ld(c)
    handle = cuda.Device().cublas_handle
    cuda.cublas.sgetrfBatched(
        handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
    cuda.cublas.sgetriBatched(
        handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,
        info.data.ptr, n_matrices)
    return c

