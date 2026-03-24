from memory import stack_allocation
from gpu.memory import AddressSpace
from gpu import thread_idx, block_idx, grid_dim, barrier
from os.atomic import Atomic
from gpu.host import DeviceContext
from math import ceildiv

comptime TBsize = 512

# level1.asum
# computes the sum of absolute values of vector elements
fn asum_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    sx: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin]
):

    var local_tid = thread_idx.x

    var shared_sums = stack_allocation[
        BLOCK,
        Scalar[dtype],
        address_space = AddressSpace.SHARED
    ]()

    var global_tid = block_idx.x * BLOCK + local_tid
    var n_threads = grid_dim.x * BLOCK

    # Each thread computes partial sum
    var local_sum = Scalar[dtype](0)

    for i in range(global_tid, n, n_threads):
        var idx = i * incx
        local_sum += abs(sx[idx])

    shared_sums[local_tid] = local_sum

    barrier()

    # Parallel reduction within block
    var stride = BLOCK // 2
    while stride > 0:
        if local_tid < stride:
            shared_sums[local_tid] += shared_sums[local_tid + stride]

        barrier()
        stride //= 2

    # Thread 0 atomically adds block result to global sum
    if local_tid == 0:
        _ = Atomic[dtype].fetch_add(result, shared_sums[0])


fn blas_asum[dtype: DType](
    n: Int,
    d_v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    incx: Int,
    d_res: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ctx: DeviceContext
) raises:
    blas_error_if(n < 1, "asum", "n", n)
    blas_error_if(incx <=0, "asum", "incx", incx)

    comptime kernel = asum_device[TBsize, dtype]
    ctx.enqueue_function[kernel, kernel](
        n,
        d_v, incx,
        d_res,
        grid_dim=ceildiv(n, TBsize),     # total thread blocks
        block_dim=TBsize                 # threads per block
    )
    ctx.synchronize()
