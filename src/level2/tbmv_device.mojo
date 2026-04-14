from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv
from ..level1.copy_device import blas_copy

comptime TBsize = 512

# level2.tbmv
# Performs triangular band matrix-vector multiplication
#    x := A*x,   or   x := A**T*x,
# where x is an n element vector and A is an n by n unit, or non-unit,
# upper or lower triangular band matrix with k super/sub-diagonals.
#
# Row-major band storage:
#   Upper: A[i,j] stored at A_band[i*lda + (j-i)], for j in [i, min(n-1, i+k)]
#   Lower: A[i,j] stored at A_band[i*lda + (i-j)], for j in [max(0, i-k), i]
#
# uplo:  0 = lower triangular, 1 = upper triangular
# trans: 0 = no transpose,     1 = transpose
# diag:  0 = non-unit diagonal, 1 = unit diagonal (diagonal not accessed)

fn stbmv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    k: Int,
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int,
    temp: UnsafePointer[Float32, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    if not trans:
        if uplo:
            # Upper triangular, no-transpose: temp[i] = sum_{j=i}^{min(n-1,i+k)} A[i,j]*x[j]
            for i in range(global_i, n, n_threads):
                var sum = Float32(0)
                var j_end = min(n - 1, i + k)
                for j in range(i, j_end + 1):
                    if j == i and diag:
                        sum += x[j * incx]
                    else:
                        sum += A[i * lda + (j - i)] * x[j * incx]
                temp[i] = sum
        else:
            # Lower triangular, no-transpose: temp[i] = sum_{j=max(0,i-k)}^{i} A[i,j]*x[j]
            for i in range(global_i, n, n_threads):
                var sum = Float32(0)
                var j_start = max(0, i - k)
                for j in range(j_start, i + 1):
                    if j == i and diag:
                        sum += x[j * incx]
                    else:
                        sum += A[i * lda + (i - j)] * x[j * incx]
                temp[i] = sum
    else:
        if uplo:
            # Upper triangular, transpose: temp[j] = sum_{i=max(0,j-k)}^{j} A[i,j]*x[i]
            for j in range(global_i, n, n_threads):
                var sum = Float32(0)
                var i_start = max(0, j - k)
                for i in range(i_start, j + 1):
                    if i == j and diag:
                        sum += x[i * incx]
                    else:
                        sum += A[i * lda + (j - i)] * x[i * incx]
                temp[j] = sum
        else:
            # Lower triangular, transpose: temp[j] = sum_{i=j}^{min(n-1,j+k)} A[i,j]*x[i]
            for j in range(global_i, n, n_threads):
                var sum = Float32(0)
                var i_end = min(n - 1, j + k)
                for i in range(j, i_end + 1):
                    if i == j and diag:
                        sum += x[i * incx]
                    else:
                        sum += A[i * lda + (i - j)] * x[i * incx]
                temp[j] = sum


fn dtbmv_device(
    uplo: Int,
    trans: Int,
    diag: Int,
    n: Int,
    k: Int,
    A: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int,
    temp: UnsafePointer[Float64, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    if not trans:
        if uplo:
            # Upper triangular, no-transpose: temp[i] = sum_{j=i}^{min(n-1,i+k)} A[i,j]*x[j]
            for i in range(global_i, n, n_threads):
                var sum = Float64(0)
                var j_end = min(n - 1, i + k)
                for j in range(i, j_end + 1):
                    if j == i and diag:
                        sum += x[j * incx]
                    else:
                        sum += A[i * lda + (j - i)] * x[j * incx]
                temp[i] = sum
        else:
            # Lower triangular, no-transpose: temp[i] = sum_{j=max(0,i-k)}^{i} A[i,j]*x[j]
            for i in range(global_i, n, n_threads):
                var sum = Float64(0)
                var j_start = max(0, i - k)
                for j in range(j_start, i + 1):
                    if j == i and diag:
                        sum += x[j * incx]
                    else:
                        sum += A[i * lda + (i - j)] * x[j * incx]
                temp[i] = sum
    else:
        if uplo:
            # Upper triangular, transpose: temp[j] = sum_{i=max(0,j-k)}^{j} A[i,j]*x[i]
            for j in range(global_i, n, n_threads):
                var sum = Float64(0)
                var i_start = max(0, j - k)
                for i in range(i_start, j + 1):
                    if i == j and diag:
                        sum += x[i * incx]
                    else:
                        sum += A[i * lda + (j - i)] * x[i * incx]
                temp[j] = sum
        else:
            # Lower triangular, transpose: temp[j] = sum_{i=j}^{min(n-1,j+k)} A[i,j]*x[i]
            for j in range(global_i, n, n_threads):
                var sum = Float64(0)
                var i_end = min(n - 1, j + k)
                for i in range(j, i_end + 1):
                    if i == j and diag:
                        sum += x[i * incx]
                    else:
                        sum += A[i * lda + (i - j)] * x[i * incx]
                temp[j] = sum


fn blas_tbmv[dtype: DType](
    uplo: Int,
    trans: Bool,
    diag: Int,
    n: Int,
    k: Int,
    d_A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lda: Int,
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    d_temp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext,
) raises:

    # NOTE: add error checking here?
    # check n > 0
    # check k >= 0
    # check lda >= k + 1
    # check incx > 0

    var trans_i = 1 if trans else 0

    # Use the caller-supplied temp buffer when provided; otherwise allocate one
    # automatically for this call.  owned_temp is only valid when d_temp is null.
    var owned_temp = ctx.enqueue_create_buffer[dtype](0 if d_temp else n)
    var work_temp = d_temp if d_temp else owned_temp.unsafe_ptr()

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[stbmv_device, stbmv_device](
            uplo, trans_i, diag,
            n, k, d_A, lda,
            d_x, incx, work_temp,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dtbmv_device, dtbmv_device](
            uplo, trans_i, diag,
            n, k, d_A, lda,
            d_x, incx, work_temp,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_tbmv: Unsupported type")

    # Copy accumulated results from temp back into x using the level-1
    # copy routine (temp has stride 1; x has stride incx).
    blas_copy[dtype](n, work_temp, 1, d_x, incx, ctx)
