from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level2.spr
# Performs symmetric rank-1 update of packed symmetric matrix:
#   A := alpha*x*x**T + A
# where A is stored in packed format (column-major).
# uplo: 0 = upper triangle, 1 = lower triangle
#
# Upper triangular packed storage (column-major):
#   AP[j*(j+1)//2 + i] = A[i,j] for 0 <= i <= j < n
#
# Lower triangular packed storage (column-major):
#   AP[j*(2*n-j+1)//2 + (i-j)] = A[i,j] for 0 <= j <= i < n
#
# Each thread handles a unique row i, updating all relevant AP entries
# for that row with no data race.
fn sspr_device(
    uplo: Int,
    n: Int,
    alpha: Float32,
    x: UnsafePointer[Float32, ImmutAnyOrigin],
    incx: Int,
    AP: UnsafePointer[Float32, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # upper triangle: AP[j*(j+1)/2 + i] for j in range [i, n)
    if not uplo:
        for i in range(global_i, n, n_threads):
            var xi = alpha * x[i * incx]
            for j in range(i, n):
                AP[j * (j + 1) // 2 + i] += xi * x[j * incx]
    # lower triangle: AP[j*(2*n-j+1)/2 + (i-j)] for j in range [0, i]
    else:
        for i in range(global_i, n, n_threads):
            var xi = alpha * x[i * incx]
            for j in range(0, i + 1):
                AP[j * (2 * n - j + 1) // 2 + (i - j)] += xi * x[j * incx]


fn dspr_device(
    uplo: Int,
    n: Int,
    alpha: Float64,
    x: UnsafePointer[Float64, ImmutAnyOrigin],
    incx: Int,
    AP: UnsafePointer[Float64, MutAnyOrigin],
):
    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # upper triangle: AP[j*(j+1)/2 + i] for j in range [i, n)
    if not uplo:
        for i in range(global_i, n, n_threads):
            var xi = alpha * x[i * incx]
            for j in range(i, n):
                AP[j * (j + 1) // 2 + i] += xi * x[j * incx]
    # lower triangle: AP[j*(2*n-j+1)/2 + (i-j)] for j in range [0, i]
    else:
        for i in range(global_i, n, n_threads):
            var xi = alpha * x[i * incx]
            for j in range(0, i + 1):
                AP[j * (2 * n - j + 1) // 2 + (i - j)] += xi * x[j * incx]


fn blas_spr[dtype: DType](
    uplo: Int,
    n: Int,
    alpha: Scalar[dtype],
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_AP: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    @parameter
    if dtype == DType.float32:
        ctx.enqueue_function[sspr_device, sspr_device](
            uplo, n,
            alpha, d_x, incx,
            d_AP,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    elif dtype == DType.float64:
        ctx.enqueue_function[dspr_device, dspr_device](
            uplo, n,
            alpha, d_x, incx,
            d_AP,
            grid_dim=ceildiv(n, TBsize),
            block_dim=TBsize,
        )
    else:
        raise Error("blas_spr: Unsupported type")

    ctx.synchronize()
