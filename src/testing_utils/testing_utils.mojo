from random import rand, seed
from math import sqrt

comptime tol32: Float32 = 1e-8
comptime tol64: Float64 = 1e-16

def generate_random_arr[
    dtype: DType,
    size:  Int
](
    a:   UnsafePointer[Scalar[dtype], MutAnyOrigin],
    min_value: Scalar[dtype],
    max_value: Scalar[dtype]
):
    # Generate random values in [0, 1]
    seed()
    rand[dtype](a, size)

    # Scale to [min, max]
    var rng = max_value - min_value
    for i in range(size):
        a[i] = min_value + a[i] * rng


def generate_random_scalar[
    dtype: DType,
](
    min_value: Scalar[dtype],
    max_value: Scalar[dtype]
) -> Scalar[dtype]:
    # Generate random values in [0, 1]
    seed()
    var result = Scalar[dtype]()
    rand[dtype](UnsafePointer(to=result), 1)

    range = max_value - min_value
    return min_value + result * range


# Error check following BLAS++ check_gemm:
# NOTE: can't get epsilon value in Mojo; using tol32, tol64
fn check_gemm_error[dtype: DType](
    m: Int, n: Int, k: Int,
    alpha: Scalar[dtype],
    beta: Scalar[dtype],
    A_norm: Scalar[dtype],
    B_norm: Scalar[dtype],
    C_ini_norm: Scalar[dtype],
    error_norm: Scalar[dtype]
) -> Bool:
    var alpha_ = max(abs(alpha), Scalar[dtype](1))
    var beta_  = max(abs(beta),  Scalar[dtype](1))
    var denom  = sqrt(Scalar[dtype](k) + Scalar[dtype](2)) * alpha_ * A_norm * B_norm
               + Scalar[dtype](2) * beta_ * C_ini_norm

    if denom == Scalar[dtype](0):
        return error_norm == Scalar[dtype](0)

    var err = error_norm / denom

    @parameter
    if dtype == DType.float32:
        return err < Scalar[dtype](tol32)
    else:
        return err < Scalar[dtype](tol64)


fn frobenius_norm[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int
) -> Scalar[dtype]:
    var sum = Scalar[dtype](0)
    for i in range(n):
        sum += a[i] * a[i]
    return sqrt(sum)
