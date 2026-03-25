from gpu import grid_dim, block_dim, global_idx
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

fn axpy_device[dtype: DType](
    n: Int,
    a: Scalar[dtype],
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int
):

    var global_i = global_idx.x
    var n_threads = Int(grid_dim.x * block_dim.x)

    # Multiple cells per thread
    for i in range(global_i, n, n_threads):
        y[i*incy] += a * x[i*incx]


fn blas_axpy[dtype: DType](
    n: Int,
    a: Scalar[dtype],
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    ctx: DeviceContext
) raises:
    blas_error_if["blas_axpy", "n < 0"](n < 0)
    blas_error_if["blas_axpy", "incx == 0"](incx == 0)
    blas_error_if["blas_axpy", "incy == 0"](incy == 0)

    # quick return
    if(a == 0) :
        return
        
    comptime kernel = axpy_device[dtype]
    ctx.enqueue_function[kernel, kernel](
        n, a,
        d_x, incx,
        d_y, incy,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
