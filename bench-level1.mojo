from gpu.host import DeviceContext
from sys import has_accelerator, argv
from time import monotonic
from src import *

# Reference: https://github.com/icl-utk-edu/blaspp/blob/master/test/run_tests.py

comptime WARMUP = 10


def bytes_per_elem(dtype: DType) -> Int:
    if dtype == DType.float32:
        return 4
    if dtype == DType.float64:
        return 8
    return 0


struct RunParams:
    var routines: List[String]
    var dtype_str: String
    var sizes: List[Int]
    var iters: Int

    fn __init__(out self):
        self.routines = List[String]()
        self.dtype_str = String("all")
        self.sizes = List[Int]()
        self.iters = 100


def parse_args(mut params: RunParams) -> Bool:
    var args = argv()
    var n_custom = 0

    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--type":
            if i + 1 < len(args):
                params.dtype_str = String(args[i + 1])
                i += 2
            else:
                print("--type requires a value")
                return False
        elif arg == "--n":
            if i + 1 < len(args):
                n_custom = Int(args[i + 1])
                i += 2
            else:
                print("--n requires a value")
                return False
        elif arg == "--iters":
            if i + 1 < len(args):
                params.iters = Int(args[i + 1])
                i += 2
            else:
                print("--iters requires a value")
                return False
        elif not arg.startswith("-"):
            params.routines.append(arg)
            i += 1
        else:
            i += 1

    if n_custom > 0:
        params.sizes.append(n_custom)
    else:
        params.sizes.append(1024)
        params.sizes.append(8192)
        params.sizes.append(1048576)
        params.sizes.append(8388608)
        params.sizes.append(16777216)

    return True


def bench_copy[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)

    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)

    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)

    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)

    for _ in range(WARMUP):
        blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)

    start = monotonic()
    for _ in range(iters):
        blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    var bw_gbs = Float64(2 * n * bytes_per_elem(dtype)) / avg

    print("copy,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def main():
    if not has_accelerator():
        print("No accelerator detected")
        return

    var params = RunParams()
    if not parse_args(params):
        return

    print("op, device, dtype, n, iters, avg time, bandwidth")

    with DeviceContext() as ctx:
        for i in range(len(params.sizes)):
            var n = params.sizes[i]
            if params.dtype_str == "float32" or params.dtype_str == "all":
                bench_copy[DType.float32](n, params.iters, ctx)
            if params.dtype_str == "float64" or params.dtype_str == "all":
                bench_copy[DType.float64](n, params.iters, ctx)
