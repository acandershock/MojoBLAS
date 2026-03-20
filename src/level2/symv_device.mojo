from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.symv
# Performs matrix-vector multiplication of form
#    y := alpha*A*x + beta*y
# where alpha and beta are scalars, x and y are vectors and A is an n by n symmetric matrix.
fn ssymv_device(
    upper: Int,
    n: Int,
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
        for j in range(n):
            var val: Float32
            if upper:
                if j >= i:
                    val = A[i * lda + j]
                else:
                    val = A[j * lda + i]
            else:
                if j <= i:
                    val = A[i * lda + j]
                else:
                    val = A[j * lda + i]
            sum += val * x[j * incx]
        y[i * incy] = alpha * sum + beta * y[i * incy]


fn dsymv_device(
    upper: Int,
    n: Int,
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
        for j in range(n):
            var val: Float64
            if upper:
                if j >= i:
                    val = A[i * lda + j]
                else:
                    val = A[j * lda + i]
            else:
                if j <= i:
                    val = A[i * lda + j]
                else:
                    val = A[j * lda + i]
            sum += val * x[j * incx]
        y[i * incy] = alpha * sum + beta * y[i * incy]


fn blas_symv[dtype: DType](
    upper: Bool,
    n: Int,
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
    # NOTE: add error checking here?

    var upper_i = 1 if upper else 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[ssymv_device, ssymv_device](
            upper_i, n,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dsymv_device, dsymv_device](
            upper_i, n,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_symv: Unsupported type")

    ctx.synchronize()
