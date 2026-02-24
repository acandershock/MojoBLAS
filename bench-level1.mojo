from gpu.host import DeviceContext
from sys import has_accelerator
from time import monotonic

# importing test wrappers
from src import *
from random import rand, seed

comptime WARMUP = 10

#
def fill_random[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int
):
    rand[dtype](a, n)

def bytes_per_elem(dtype: DType) -> Int:
    if dtype == DType.float16:
        return 2
    if dtype == DType.float32:
        return 4
    if dtype == DType.float64:
        return 8
    return 0

#
def bench_copy[dtype: DType](n: Int, iters: Int):
    with DeviceContext() as ctx:
        x_h = ctx.enqueue_create_host_buffer[dtype](n)
        y_h = ctx.enqueue_create_host_buffer[dtype](n)

        fill_random[dtype](x_h.unsafe_ptr(), n)
        fill_random[dtype](y_h.unsafe_ptr(), n)

        x_d = ctx.enqueue_create_buffer[dtype](n)
        y_d = ctx.enqueue_create_buffer[dtype](n)

        ctx.enqueue_copy(x_d, x_h)
        ctx.enqueue_copy(y_d, y_h)

        for _ in range(WARMUP):
            blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
        ctx.synchronize()

        total: UInt = 0
        start = monotonic()
        for _ in range (iters):
            blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
        ctx.synchronize()
        end = monotonic()
        total += (end - start)

        avg = total / iters

        var elem_bytes = bytes_per_elem(dtype)
        var bytes_per_call: Float64 = Float64(2 * n * elem_bytes)
        var avg_f: Float64 = Float64(avg)
        var bw_gbs = bytes_per_call / avg_f

        print("copy mojo, ", ctx.name(), ", ", dtype.__str__(), ", ", n, ", ", iters, ", ", avg, ",", bw_gbs)

def main():
    if not has_accelerator():
        print("No accelerator detected")
        return

    print("op, backend, gpu, dtype, N, iters, avg time (nanoseconds)\n")
    bench_copy[DType.float16](1048576, iters=1000)
    bench_copy[DType.float16](8388608, iters=500)
    bench_copy[DType.float16](16777216, iters=200)
    print("\n")

    bench_copy[DType.float32](1048576, iters=1000)
    bench_copy[DType.float32](8388608, iters=500)
    bench_copy[DType.float32](16777216, iters=200)
    print("\n")

    bench_copy[DType.float64](1048576, iters=1000)
    bench_copy[DType.float64](8388608, iters=500)
    bench_copy[DType.float64](16777216, iters=200)
    print("\n")