from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.sbmv
# Performs symmetric band matrix-vector multiplication
#    y := alpha*A*x + beta*y,
# where A is an n by n symmetric band matrix with k off-diagonals.

fn ssbmv_device(
    uplo: Int,
    n: Int,
    k: Int,
    alpha: Float32,
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int,
    beta: Float32,
    y: UnsafePointer[Float32, MutAnyOrigin],
    incy: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    for i in range(global_i, n, n_threads):
        var sum = Scalar[DType.float32](0)

        var j_start = max(0, i - k)
        var j_end = min(n - 1, i + k)

        for j in range(j_start, j_end + 1):
            var val: Float32

            if uplo: # upper
                if j >= i:
                    val = A[i * lda + (j - i)]
                else:
                    val = A[j * lda + (i - j)]
            else: # lower
                if j <= i:
                    val = A[i * lda + (i - j)]
                else:
                    val = A[j * lda + (j - i)]

            sum += val * x[j * incx]

        y[i * incy] = alpha * sum + beta * y[i * incy]


fn dsbmv_device(
    uplo: Int,
    n: Int,
    k: Int,
    alpha: Float64,
    A: UnsafePointer[Float64, ImmutAnyOrigin],
    lda: Int,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int,
    beta: Float64,
    y: UnsafePointer[Float64, MutAnyOrigin],
    incy: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    for i in range(global_i, n, n_threads):
        var sum = Scalar[DType.float64](0)

        var j_start = max(0, i - k)
        var j_end = min(n - 1, i + k)

        for j in range(j_start, j_end + 1):
            var val: Float64

            if uplo: # upper
                if j >= i:
                    val = A[i * lda + (j - i)]
                else:
                    val = A[j * lda + (i - j)]
            else: # lower
                if j <= i:
                    val = A[i * lda + (i - j)]
                else:
                    val = A[j * lda + (j - i)]

            sum += val * x[j * incx]

        y[i * incy] = alpha * sum + beta * y[i * incy]


fn blas_sbmv[dtype: DType](
    uplo: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    d_A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lda: Int,
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    beta: Scalar[dtype],
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    ctx: DeviceContext,
) raises:

    # TODO:
    # check n > 0
    # check k >= 0
    # check lda >= k + 1
    # check incx, incy > 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[ssbmv_device, ssbmv_device](
            uplo, n, k,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dsbmv_device, dsbmv_device](
            uplo, n, k,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_sbmv: Unsupported type")

    ctx.synchronize()
