from gpu import thread_idx, block_idx, block_dim, lane_id
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from gpu.primitives.warp import sum as warp_sum, WARP_SIZE
from math import ceildiv,sqrt
from buffer import NDBuffer, DimList
from algorithm import sum
from layout import Layout, LayoutTensor
from os.atomic import Atomic

fn nrm2_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    shared_scratch = stack_allocation[
        BLOCK,
        Float64,
        address_space = AddressSpace.SHARED
    ]()

    # Each thread computes one partial square
    var local_sum = Scalar[DType.float64](0)
    if global_i < UInt(n):
        v = x[global_i * incx]
        vf64 = v.cast[DType.float64]()
        
        local_sum += vf64 * vf64

    shared_scratch[local_i] = local_sum

    barrier()

    var stride = BLOCK // 2
    while stride > 0:
        if local_i < stride:
            shared_scratch[local_i] += shared_scratch[local_i + stride]
        barrier()
        stride //= 2

    # Lane 0 accumulates into global output
    if local_i == 0:
        real_total = Scalar[dtype](shared_scratch[0])
        _ = Atomic[dtype].fetch_add(output, real_total)


fn blas_nrm2[dtype: DType](
    n: Int,
    d_x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext
) raises:
    comptime kernel = nrm2_device[TBsize, dtype]
    ctx.enqueue_function[kernel, kernel](
        n, d_x, incx, d_out,
        grid_dim=ceildiv(n, TBsize),
        block_dim=TBsize,
    )
    ctx.synchronize()
    
