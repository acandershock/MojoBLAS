from gpu import thread_idx, block_idx, block_dim, grid_dim
from os.atomic import Atomic
from memory import stack_allocation
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level1.dot
# computes the dot product of two vectors
fn dot_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):

    var global_i = block_dim.x * block_idx.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x
    var local_i = thread_idx.x

    shared_res = stack_allocation[
        BLOCK,
        Float64,
        address_space = AddressSpace.SHARED
    ]()

    var thread_sum = Scalar[dtype](0)
    for i in range(global_i, n, n_threads):
        thread_sum += x[i * incx] * y[i * incy]

    shared_res[local_i] = Float64(thread_sum)
    barrier()

    var stride = BLOCK // 2
    while stride > 0:
        if local_i < stride:
            shared_res[local_i] += shared_res[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        _ = Atomic[dtype].fetch_add(output, Scalar[dtype](shared_res[0]))


fn blas_dot[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incy: Int,
    d_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext
) raises:
    
    blas_error_if(n < 1, "blas_dot", "n", n)

    comptime kernel = dot_device[TBsize, dtype]
    ctx.enqueue_function[kernel, kernel](
        n, d_x, incx,
        d_y, incy, d_out,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
