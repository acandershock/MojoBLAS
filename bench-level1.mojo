from gpu.host import DeviceContext
from sys import has_accelerator, argv
from time import monotonic
from math import sin, cos
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

    if len(params.routines) == 0:
        # TODO: add rotm, rotmg
        params.routines = ["asum", "axpy", "copy", "dot", "dotc",
                           "dotu", "iamax", "nrm2", "rot", "rotg", "scal", "swap"]

    return True


def bench_asum[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_asum[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    start = monotonic()
    for _ in range(iters):
        blas_asum[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: n reads
    var bw_gbs = Float64(n * bytes_per_elem(dtype)) / avg
    print("asum,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_axpy[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](2.0)

    for _ in range(WARMUP):
        blas_axpy[dtype](n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)

    start = monotonic()
    for _ in range(iters):
        blas_axpy[dtype](n, alpha, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: 2n reads + n writes = 3n
    var bw_gbs = Float64(3 * n * bytes_per_elem(dtype)) / avg
    print("axpy,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_copy[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)

    start = monotonic()
    for _ in range(iters):
        blas_copy[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: n reads + n writes = 2n
    var bw_gbs = Float64(2 * n * bytes_per_elem(dtype)) / avg
    print("copy,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_dot[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_dot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    start = monotonic()
    for _ in range(iters):
        blas_dot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: 2n reads
    var bw_gbs = Float64(2 * n * bytes_per_elem(dtype)) / avg
    print("dot,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_dotc[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    y_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    generate_random_arr[dtype](2 * n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](2 * n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](2 * n)
    y_d = ctx.enqueue_create_buffer[dtype](2 * n)
    res_d = ctx.enqueue_create_buffer[dtype](2)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_dotc[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    start = monotonic()
    for _ in range(iters):
        blas_dotc[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: 2 vectors * 2n floats = 4n reads
    var bw_gbs = Float64(4 * n * bytes_per_elem(dtype)) / avg
    print("dotc,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_dotu[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    y_h = ctx.enqueue_create_host_buffer[dtype](2 * n)
    generate_random_arr[dtype](2 * n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](2 * n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](2 * n)
    y_d = ctx.enqueue_create_buffer[dtype](2 * n)
    res_d = ctx.enqueue_create_buffer[dtype](2)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_dotu[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    start = monotonic()
    for _ in range(iters):
        blas_dotu[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: 2 vectors * 2n floats = 4n reads
    var bw_gbs = Float64(4 * n * bytes_per_elem(dtype)) / avg
    print("dotu,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_iamax[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[DType.int64](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_iamax[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    start = monotonic()
    for _ in range(iters):
        blas_iamax[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: n reads
    var bw_gbs = Float64(n * bytes_per_elem(dtype)) / avg
    print("iamax,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_nrm2[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    res_d = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_nrm2[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)

    start = monotonic()
    for _ in range(iters):
        blas_nrm2[dtype](n, x_d.unsafe_ptr(), 1, res_d.unsafe_ptr(), ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: n reads
    var bw_gbs = Float64(n * bytes_per_elem(dtype)) / avg
    print("nrm2,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_rot[dtype: DType](n: Int, iters: Int, ctx: DeviceContext) where dtype.is_floating_point():
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var angle = generate_random_scalar[dtype](0, 2 * 3.14159265359)
    var c = Scalar[dtype](cos(angle))
    var s = Scalar[dtype](sin(angle))

    for _ in range(WARMUP):
        blas_rot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, c, s, ctx)

    start = monotonic()
    for _ in range(iters):
        blas_rot[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, c, s, ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: 2n reads + 2n writes = 4n
    var bw_gbs = Float64(4 * n * bytes_per_elem(dtype)) / avg
    print("rot,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_rotg[dtype: DType](iters: Int):
    var a = generate_random_scalar[dtype](-100, 100)
    var b = generate_random_scalar[dtype](-100, 100)
    var c = Scalar[dtype](0)
    var s = Scalar[dtype](0)

    for _ in range(WARMUP):
        blas_rotg[dtype](UnsafePointer(to=a), UnsafePointer(to=b), UnsafePointer(to=c), UnsafePointer(to=s))

    start = monotonic()
    for _ in range(iters):
        blas_rotg[dtype](UnsafePointer(to=a), UnsafePointer(to=b), UnsafePointer(to=c), UnsafePointer(to=s))
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    print("rotg, cpu,", dtype, ", -, ", iters, ",", Int(avg), "ns")


# TODO: uncomment once rotmg is added
# def bench_rotm[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
#     x_h = ctx.enqueue_create_host_buffer[dtype](n)
#     y_h = ctx.enqueue_create_host_buffer[dtype](n)
#     generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
#     generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
#     x_d = ctx.enqueue_create_buffer[dtype](n)
#     y_d = ctx.enqueue_create_buffer[dtype](n)
#     ctx.enqueue_copy(x_d, x_h)
#     ctx.enqueue_copy(y_d, y_h)

#     # d1 and d2 must be positive
#     var d1 = generate_random_scalar[dtype](1, 100)
#     var d2 = generate_random_scalar[dtype](1, 100)
#     var x1 = generate_random_scalar[dtype](-100, 100)
#     var y1 = generate_random_scalar[dtype](-100, 100)
#     param_h = ctx.enqueue_create_host_buffer[dtype](5)
#     param_d = ctx.enqueue_create_buffer[dtype](5)
#     # NOTE: need rotmg to compute a valid param
#     ctx.enqueue_copy(param_d, param_h)
#     ctx.synchronize()

#     for _ in range(WARMUP):
#         blas_rotm[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, param_d.unsafe_ptr(), ctx)

#     start = monotonic()
#     for _ in range(iters):
#         blas_rotm[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, param_d.unsafe_ptr(), ctx)
#     end = monotonic()

#     var avg = Float64(end - start) / Float64(iters)
#     # bandwidth: 2n reads + 2n writes = 4n
#     var bw_gbs = Float64(4 * n * bytes_per_elem(dtype)) / avg
#     print("rotm,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


# TODO: add bench_rotmg


def bench_scal[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.synchronize()

    var alpha = Scalar[dtype](2.0)

    for _ in range(WARMUP):
        blas_scal[dtype](n, alpha, x_d.unsafe_ptr(), 1, ctx)

    start = monotonic()
    for _ in range(iters):
        blas_scal[dtype](n, alpha, x_d.unsafe_ptr(), 1, ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: n reads + n writes = 2n
    var bw_gbs = Float64(2 * n * bytes_per_elem(dtype)) / avg
    print("scal,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def bench_swap[dtype: DType](n: Int, iters: Int, ctx: DeviceContext):
    x_h = ctx.enqueue_create_host_buffer[dtype](n)
    y_h = ctx.enqueue_create_host_buffer[dtype](n)
    generate_random_arr[dtype](n, x_h.unsafe_ptr(), -1000, 1000)
    generate_random_arr[dtype](n, y_h.unsafe_ptr(), -1000, 1000)
    x_d = ctx.enqueue_create_buffer[dtype](n)
    y_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    for _ in range(WARMUP):
        blas_swap[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)

    start = monotonic()
    for _ in range(iters):
        blas_swap[dtype](n, x_d.unsafe_ptr(), 1, y_d.unsafe_ptr(), 1, ctx)
    end = monotonic()

    var avg = Float64(end - start) / Float64(iters)
    # bandwidth: 2n reads + 2n writes = 4n
    var bw_gbs = Float64(4 * n * bytes_per_elem(dtype)) / avg
    print("swap,", ctx.name(), ",", dtype, ",", n, ",", iters, ",", Int(avg), "ns,", bw_gbs, "GB/s")


def run_dtype[
    dtype: DType
](
    routine: String,
    params: RunParams,
    ctx: DeviceContext
) where dtype.is_floating_point():
    for i in range(len(params.sizes)):
        var n = params.sizes[i]
        if   (routine == "asum"):  bench_asum[dtype](n, params.iters, ctx)
        elif (routine == "axpy"):  bench_axpy[dtype](n, params.iters, ctx)
        elif (routine == "copy"):  bench_copy[dtype](n, params.iters, ctx)
        elif (routine == "dot"):   bench_dot[dtype](n, params.iters, ctx)
        elif (routine == "dotc"):  bench_dotc[dtype](n, params.iters, ctx)
        elif (routine == "dotu"):  bench_dotu[dtype](n, params.iters, ctx)
        elif (routine == "iamax"): bench_iamax[dtype](n, params.iters, ctx)
        elif (routine == "nrm2"):  bench_nrm2[dtype](n, params.iters, ctx)
        elif (routine == "rot"):   bench_rot[dtype](n, params.iters, ctx)
        elif (routine == "rotg"):
            bench_rotg[dtype](params.iters)
            return
        # elif (routine == "rotm"):  bench_rotm[dtype](n, params.iters, ctx)
        # elif (routine == "rotmg"):
            # bench_rotmg[dtype](params.iters)  # TODO: implement blas_rotmg
            # return
        elif (routine == "scal"):  bench_scal[dtype](n, params.iters, ctx)
        elif (routine == "swap"):  bench_swap[dtype](n, params.iters, ctx)
        else:
            print("Unknown routine:", routine, "for", dtype)
            return


def main():
    if not has_accelerator():
        print("No accelerator detected")
        return

    var params = RunParams()
    if not parse_args(params):
        return

    print("op, device, dtype, n, iters, avg time, bandwidth")

    with DeviceContext() as ctx:
        for routine in(params.routines):
            if params.dtype_str == "float32" or params.dtype_str == "all":
                run_dtype[DType.float32](routine, params, ctx)

            if params.dtype_str == "float64" or params.dtype_str == "all":
                run_dtype[DType.float64](routine, params, ctx)
