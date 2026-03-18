from random import rand, seed
from math import sqrt

from python import Python

def generate_random_arr[
    dtype: DType
](
    size:  Int,
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
) raises -> Bool:
    np = Python.import_module("numpy")
    tol32 = Scalar[DType.float32](py=np.finfo(np.float32).eps)
    tol64 = Scalar[DType.float64](py=np.finfo(np.float64).eps)

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

fn check_syr_error[dtype: DType](
    n: Int,
    alpha: Scalar[dtype],
    x_norm: Scalar[dtype],
    y_norm: Scalar[dtype],
    A_ini_norm: Scalar[dtype],
    error_norm: Scalar[dtype]
) raises -> Bool:
    np = Python.import_module("numpy")
    tol32 = Scalar[DType.float32](py=np.finfo(np.float32).eps)
    tol64 = Scalar[DType.float64](py=np.finfo(np.float64).eps)

    var alpha_ = max(abs(alpha), Scalar[dtype](1))

    var denom =
        Scalar[dtype](2) * alpha_ * x_norm * y_norm +
        Scalar[dtype](2) * A_ini_norm

    if denom == Scalar[dtype](0):
        return error_norm == Scalar[dtype](0)

    var err = error_norm / denom

    @parameter
    if dtype == DType.float32:
        return err < Scalar[dtype](tol32)
    else:
        return err < Scalar[dtype](tol64)

fn frobenius_norm_symmetric[dtype: DType](
    C: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ldc: Int,
    lower: Int # 0 = upper triangle, 1 = lower triangle
) -> Scalar[dtype]:

    var sum = Scalar[dtype](0)

    if lower == 1:
        for j in range(n):
            for i in range(j, n):
                var val = C[i + j*ldc]
                if i == j:
                    sum += val * val
                else:
                    sum += Scalar[dtype](2) * val * val
    else:
        for j in range(n):
            for i in range(j+1):
                var val = C[i + j*ldc]
                if i == j:
                    sum += val * val
                else:
                    sum += Scalar[dtype](2) * val * val


    return sqrt(sum)

fn dense_to_band[dtype: DType](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    m: Int,
    n: Int,
    kl: Int,
    ku: Int
):
    band_width = kl + ku + 1

    for i in range(m):
        for j in range(band_width):
            B[i * band_width + j] = 0.0

        for j in range(n):
            if j >= i - kl and j <= i + ku:
                band_col = j - (i - kl)
                B[i * band_width + band_col] = A[i * n + j]
            else:
                A[i * n + j] = Scalar[dtype](0)

fn dense_to_sym_band[dtype: DType](
    A_dense: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A_band: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    k: Int,
    upper: Bool,
):
    var lda = k + 1

    # zero initialize
    for i in range(n):
        for b in range(lda):
            A_band[i * lda + b] = 0

    if upper:
        # store A[i,j] for j >= i
        for i in range(n):
            var j_end = (i + k) if (i + k < n - 1) else (n - 1)
            for j in range(i, j_end + 1):
                var band_col = j - i
                A_band[i * lda + band_col] = A_dense[i * n + j]
    else:
        # store A[i,j] for j <= i
        for i in range(n):
            var j_start = (i - k) if (i - k > 0) else 0
            for j in range(j_start, i + 1):
                var band_col = i - j
                A_band[i * lda + band_col] = A_dense[i * n + j]

    # Overwrite original matrix with band reconstruction
    for i in range(n):
        for j in range(n):
            var val = Scalar[dtype](0)

            if upper:
                if j >= i and j <= i + k:
                    val = A_band[i * lda + (j - i)]
                elif i >= j and i <= j + k:
                    # symmetric mirror
                    val = A_band[j * lda + (i - j)]
            else:
                if j <= i and j >= i - k:
                    val = A_band[i * lda + (i - j)]
                elif j >= i and j <= i + k:
                    # symmetric mirror
                    val = A_band[j * lda + (j - i)]

            A_dense[i * n + j] = val
