from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext

# level2.tbsv
# Solves a triangular band system
#    A*x = b,   or   A**T*x = b,
# where b and x are n element vectors and A is an n by n unit, or non-unit,
# upper or lower triangular band matrix with k super/sub-diagonals.
#
# Row-major band storage (same layout as tbmv):
#   Upper: A[i,j] stored at A_band[i*lda + (j-i)], for j in [i, min(n-1, i+k)]
#   Lower: A[i,j] stored at A_band[i*lda + (i-j)], for j in [max(0, i-k), i]
#   Diagonal A[i,i] is always at A_band[i*lda + 0] for both upper and lower.
#
# uplo:  0 = lower triangular, 1 = upper triangular
# trans: 0 = no transpose,     1 = transpose
# diag:  0 = non-unit diagonal, 1 = unit diagonal (diagonal not accessed)

fn stbsv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    k: Int,
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
            # Lower triangular, no-transpose: forward substitution
            # A[i,j] at A_band[i*lda + (i-j)]; diagonal at A_band[i*lda + 0]
            for i in range(n):
                var temp = x[i * incx]
                var j_start = max(0, i - k)
                for j in range(j_start, i):
                    temp -= A[i * lda + (i - j)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp
        else:
            # Upper triangular, no-transpose: backward substitution
            # A[i,j] at A_band[i*lda + (j-i)]; diagonal at A_band[i*lda + 0]
            for idx in range(n):
                var i = n - 1 - idx
                var temp = x[i * incx]
                var j_end = min(n - 1, i + k)
                for j in range(i + 1, j_end + 1):
                    temp -= A[i * lda + (j - i)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp
    else:
        if not uplo:
            # Transpose lower triangular (A^T is upper): backward substitution
            # A^T[i,j] = A[j,i]; for lower A, A[j,i] at A_band[j*lda + (j-i)]
            # where j in [i+1, min(n-1, i+k)]
            for idx in range(n):
                var i = n - 1 - idx
                var temp = x[i * incx]
                var j_end = min(n - 1, i + k)
                for j in range(i + 1, j_end + 1):
                    temp -= A[j * lda + (j - i)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp
        else:
            # Transpose upper triangular (A^T is lower): forward substitution
            # A^T[i,j] = A[j,i]; for upper A, A[j,i] at A_band[j*lda + (i-j)]
            # where j in [max(0, i-k), i-1]
            for i in range(n):
                var temp = x[i * incx]
                var j_start = max(0, i - k)
                for j in range(j_start, i):
                    temp -= A[j * lda + (i - j)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp


fn dtbsv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    k: Int,
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
            # Lower triangular, no-transpose: forward substitution
            for i in range(n):
                var temp = x[i * incx]
                var j_start = max(0, i - k)
                for j in range(j_start, i):
                    temp -= A[i * lda + (i - j)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp
        else:
            # Upper triangular, no-transpose: backward substitution
            for idx in range(n):
                var i = n - 1 - idx
                var temp = x[i * incx]
                var j_end = min(n - 1, i + k)
                for j in range(i + 1, j_end + 1):
                    temp -= A[i * lda + (j - i)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp
    else:
        if not uplo:
            # Transpose lower triangular (A^T is upper): backward substitution
            for idx in range(n):
                var i = n - 1 - idx
                var temp = x[i * incx]
                var j_end = min(n - 1, i + k)
                for j in range(i + 1, j_end + 1):
                    temp -= A[j * lda + (j - i)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp
        else:
            # Transpose upper triangular (A^T is lower): forward substitution
            for i in range(n):
                var temp = x[i * incx]
                var j_start = max(0, i - k)
                for j in range(j_start, i):
                    temp -= A[j * lda + (i - j)] * x[j * incx]
                if not diag:
                    temp /= A[i * lda]
                x[i * incx] = temp


fn blas_tbsv[dtype: DType](
    uplo: Int,
    trans: Bool,
    diag: Int,
    n: Int,
    k: Int,
    d_A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lda: Int,
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    ctx: DeviceContext,
) raises:

    # NOTE: add error checking here?
    # check n > 0
    # check k >= 0
    # check lda >= k + 1
    # check incx > 0

    var trans_i = 1 if trans else 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[stbsv_device, stbsv_device](
            uplo, trans_i, diag,
            n, k, d_A, lda,
            d_x, incx,
            grid_dim=1,
            block_dim=1,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dtbsv_device, dtbsv_device](
            uplo, trans_i, diag,
            n, k, d_A, lda,
            d_x, incx,
            grid_dim=1,
            block_dim=1,
        )
    else:
        raise Error("blas_tbsv: Unsupported type")

    ctx.synchronize()
