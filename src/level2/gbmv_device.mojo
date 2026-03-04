from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.gbmv
# Performs band matrix-vector multiplication of form
#    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
# where A is an m by n band matrix with kl sub-diagonals and ku super-diagonals.

fn sgbmv_device(
    trans: Int,
    m: Int,
    n: Int,
    kl: Int,
    ku: Int,
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
        # y(i) = alpha * sum_j A(i,j)*x(j) + beta*y(i)
        for i in range(global_i, m, n_threads):
            var sum = Scalar[DType.float32](0)

            var j_start = (i - kl) if (i - kl > 0) else 0
            var j_end   = (i + ku) if (i + ku < n-1) else n-1

            for j in range(j_start, j_end+1):
                var band_col = kl + j - i
                sum += A[i * lda + band_col] * x[j * incx]

            y[i * incy] = alpha * sum + beta * y[i * incy]

    else:
        # y(j) = alpha * sum_i A(i,j)*x(i) + beta*y(j)
        for j in range(global_i, n, n_threads):
            var sum = Scalar[DType.float32](0)

            var i_start = (j - ku) if (j - ku > 0) else 0
            var i_end   = (j + kl) if (j + kl < m-1) else m-1

            for i in range(i_start, i_end+1):
                var band_col = kl + j - i
                sum += A[i * lda + band_col] * x[i * incx]

            y[j * incy] = alpha * sum + beta * y[j * incy]


fn dgbmv_device(
    trans: Int,
    m: Int,
    n: Int,
    kl: Int,
    ku: Int,
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
        # y(i) = alpha * sum_j A(i,j)*x(j) + beta*y(i)
        for i in range(global_i, m, n_threads):
            var sum = Scalar[DType.float64](0)

            var j_start = (i - kl) if (i - kl > 0) else 0
            var j_end   = (i + ku) if (i + ku < n-1) else n-1

            for j in range(j_start, j_end+1):
                var band_col = kl + j - i
                sum += A[i * lda + band_col] * x[j * incx]

            y[i * incy] = alpha * sum + beta * y[i * incy]

    else:
        # y(j) = alpha * sum_i A(i,j)*x(i) + beta*y(j)
        for j in range(global_i, n, n_threads):
            var sum = Scalar[DType.float64](0)

            var i_start = (j - ku) if (j - ku > 0) else 0
            var i_end   = (j + kl) if (j + kl < m-1) else m-1

            for i in range(i_start, i_end+1):
                var band_col = kl + j - i
                sum += A[i * lda + band_col] * x[i * incx]

            y[j * incy] = alpha * sum + beta * y[j * incy]


fn blas_gbmv[dtype: DType](
    trans: Bool,
    m: Int,
    n: Int,
    kl: Int,
    ku: Int,
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
    # check kl, ku >= 0
    # check lda >= kl + ku + 1
    # check incx, incy > 0

    var out_len = m if not trans else n
    var trans_i = 1 if trans else 0

    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[sgbmv_device, sgbmv_device](
            trans_i, m, n, kl, ku,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(out_len, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dgbmv_device, dgbmv_device](
            trans_i, m, n, kl, ku,
            alpha, d_A, lda,
            d_x, incx,
            beta, d_y, incy,
            grid_dim=ceildiv(out_len, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_gbmv: Unsupported type")

    ctx.synchronize()
