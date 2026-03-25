from gpu import grid_dim, block_dim, global_idx
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

fn scal_device[dtype: DType](
    n: Int,
    a: Scalar[dtype],
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
):

    var global_i = global_idx.x
    var n_threads = Int(grid_dim.x * block_dim.x)

    # Multiple cells per thread
    for i in range(global_i, n, n_threads):
        x[i*incx] *= a


fn blas_scal[dtype: DType] (
    n: Int,
    a: Scalar[dtype],
    d_x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    ctx: DeviceContext
) raises:

    blas_error_if["blas_scal", "n < 0"](n < 0)
    blas_error_if["blas_scal", "incx <= 0"](incx <= 0)


    comptime kernel = scal_device[dtype]
    ctx.enqueue_function[kernel, kernel](
        n, a, d_x, incx,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
