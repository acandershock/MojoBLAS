from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.gemv
# Performs matrix-vector multiplication of form
#    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
# where alpha and beta are scalars, x and y are vectors and A is an m by n matrix.
fn sgemv_device(
    trans: Int,
    m: Int,
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

    if not trans:
        for i in range(global_i, m, n_threads):
            var sum = Scalar[DType.float32](0)
            for j in range(n):
                sum += A[i * lda + j] * x[j * incx]
            y[i * incy] = alpha * sum + beta * y[i * incy]
    else:
        for j in range(global_i, n, n_threads):
            var sum = Scalar[DType.float32](0)
            for i in range(m):
                sum += A[i * lda + j] * x[i * incx]
            y[j * incy] = alpha * sum + beta * y[j * incy]


fn dgemv_device(
    trans: Int,
    m: Int,
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

    if not trans:
        for i in range(global_i, m, n_threads):
            var sum = Scalar[DType.float64](0)
            for j in range(n):
                sum += A[i * lda + j] * x[j * incx]
            y[i * incy] = alpha * sum + beta * y[i * incy]
    else:
        for j in range(global_i, n, n_threads):
            var sum = Scalar[DType.float64](0)
            for i in range(m):
                sum += A[i * lda + j] * x[i * incx]
            y[j * incy] = alpha * sum + beta * y[j * incy]


fn blas_gemv[dtype: DType](
    trans: Bool,
    m: Int,
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
    # check m, n > 0
    # check incx, incy > 0

    var out_len = m if not trans else n

    # Can't pass a Bool to GPU kernel
    var trans_i = 1 if trans else 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[sgemv_device, sgemv_device](
            trans_i, m, n,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(out_len, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dgemv_device, dgemv_device](
            trans_i, m, n,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(out_len, TBsize),
            block_dim=TBsize,
        )
    else:
        return

    ctx.synchronize()
