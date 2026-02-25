from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.syr2
# Performs symmetric rank-2 update:
# A := alpha*x*y**T + alpha*y*x**T + A
# uplo: 0 = upper triangle, 1 = lower triangle
fn ssyr2_device(
    uplo: Int,
    n: Int,
    alpha: Float32,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Float32, ImmutAnyOrigin],
    incy: Int,
    A: UnsafePointer[Float32, MutAnyOrigin],
    lda: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # upper triangle: update A[i,j] for j in range [i, n)
    if not uplo:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(i, n):
                A[i * lda + j] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]
    # lower triangle: update A[i,j] for j in range [0, i]
    else:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(0, i + 1):
                A[i * lda + j] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]

fn dsyr2_device(
    uplo: Int,
    n: Int,
    alpha: Float64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Float64, ImmutAnyOrigin],
    incy: Int,
    A: UnsafePointer[Float64, MutAnyOrigin],
    lda: Int,
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # upper triangle: update A[i,j] for j in range [i, n)
    if not uplo:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(i, n):
                A[i * lda + j] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]
    # lower triangle: update A[i,j] for j in range [0, i]
    else:
        for i in range(global_i, n, n_threads):
            var xi = x[i * incx]
            var yi = y[i * incy]
            var alpha_xi = alpha * xi
            var alpha_yi = alpha * yi
            for j in range(0, i + 1):
                A[i * lda + j] += alpha_xi * y[j * incy] + alpha_yi * x[j * incx]

fn blas_syr2[dtype: DType](
    uplo: Int,
    n: Int,
    alpha: Scalar[dtype],
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    d_A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lda: Int,
    ctx: DeviceContext,
) raises:
    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[ssyr2_device, ssyr2_device](
            uplo, n,
            alpha,
            d_x, incx,
            d_y, incy,
            d_A, lda,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dsyr2_device, dsyr2_device](
            uplo, n,
            alpha,
            d_x, incx,
            d_y, incy,
            d_A, lda,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_syr2: Unsupported type")

    ctx.synchronize()
