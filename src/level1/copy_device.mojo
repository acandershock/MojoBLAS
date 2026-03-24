from gpu import grid_dim, block_dim, global_idx
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level1.copy
# copies values of vector x to vector y
fn copy_device[dtype: DType](
    n: Int,
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int
):

    var global_i = global_idx.x
    var n_threads = Int(grid_dim.x * block_dim.x)

    # Multiple cells per thread
    for i in range(global_i, n, n_threads):
        y[i*incy] = x[i*incx]


fn blas_copy[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    ctx: DeviceContext
) raises:

    blas_error_if(n<=0, "blas_copy", "n", n)
    blas_error_if(incx == 0, "blas_copy", "incx", incx)
    blas_error_if(incy == 0, "blas_copy", "incy", incy)

    
    comptime kernel = copy_device[dtype]
    ctx.enqueue_function[kernel, kernel](
        n,
        d_x, incx,
        d_y, incy,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
