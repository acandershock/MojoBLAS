from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

# level2.trsv
# Triangular matrix-vector solve
# Solves one of the systems of equations
#    A*x = b,   or   A**T*x = b,
# where b and x are n element vectors and A is an n by n unit, or
# non-unit, upper or lower triangular matrix.
#
# uplo:  0 = lower triangular, 1 = upper triangular
# trans: 0 = no transpose,     1 = transpose
# diag:  0 = non-unit diagonal, 1 = unit diagonal
fn strsv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float32, MutAnyOrigin],
    incx: Int,
):
    var tid = block_dim.x * block_idx.x + thread_idx.x
    if tid != 0:
        return

    if not trans:
        if not uplo:
            # Lower triangular: forward substitution
            for i in range(n):
                var temp = x[i * incx]
                for j in range(i):
                    temp -= A[i * lda + j] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp
        else:
            # Upper triangular: backward substitution
            for k in range(n):
                var i = n - 1 - k
                var temp = x[i * incx]
                for j in range(i + 1, n):
                    temp -= A[i * lda + j] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp
    else:
        if not uplo:
            # Transpose lower triangular (A^T is upper): backward substitution
            for k in range(n):
                var i = n - 1 - k
                var temp = x[i * incx]
                for j in range(i + 1, n):
                    temp -= A[j * lda + i] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp
        else:
            # Transpose upper triangular (A^T is lower): forward substitution
            for i in range(n):
                var temp = x[i * incx]
                for j in range(i):
                    temp -= A[j * lda + i] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp


fn dtrsv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    A: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float64, MutAnyOrigin],
    incx: Int,
):
    var tid = block_dim.x * block_idx.x + thread_idx.x
    if tid != 0:
        return

    if not trans:
        if not uplo:
            # Lower triangular: forward substitution
            for i in range(n):
                var temp = x[i * incx]
                for j in range(i):
                    temp -= A[i * lda + j] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp
        else:
            # Upper triangular: backward substitution
            for k in range(n):
                var i = n - 1 - k
                var temp = x[i * incx]
                for j in range(i + 1, n):
                    temp -= A[i * lda + j] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp
    else:
        if not uplo:
            # Transpose lower triangular (A^T is upper): backward substitution
            for k in range(n):
                var i = n - 1 - k
                var temp = x[i * incx]
                for j in range(i + 1, n):
                    temp -= A[j * lda + i] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp
        else:
            # Transpose upper triangular (A^T is lower): forward substitution
            for i in range(n):
                var temp = x[i * incx]
                for j in range(i):
                    temp -= A[j * lda + i] * x[j * incx]
                if not diag:
                    temp /= A[i * lda + i]
                x[i * incx] = temp


fn blas_trsv[dtype: DType](
    uplo: Int,
    trans: Bool,
    diag: Int,
    n: Int,
    d_A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lda: Int,
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    ctx: DeviceContext,
) raises:

    # NOTE: add error checking here?
    # check n > 0
    # check incx > 0

    var trans_i = 1 if trans else 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[strsv_device, strsv_device](
            uplo, trans_i, diag,
            n, d_A, lda,
            d_x, incx,
            grid_dim=1,
            block_dim=1,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dtrsv_device, dtrsv_device](
            uplo, trans_i, diag,
            n, d_A, lda,
            d_x, incx,
            grid_dim=1,
            block_dim=1,
        )
    else:
        raise Error("blas_trsv: Unsupported type")

    ctx.synchronize()
