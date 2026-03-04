from testing import assert_equal, assert_almost_equal, assert_true, TestSuite
from gpu.host import DeviceContext

from src import *
from python import Python, PythonObject

comptime atol = 1.0E-4


def gemv_test[
    dtype: DType,
    m: Int,
    n: Int,
    trans: Bool,
]():
    # x_len and y_len depend on transpose:
    comptime x_len = n if not trans else m
    comptime y_len = m if not trans else n

    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](m * n)
        A = ctx.enqueue_create_host_buffer[dtype](m * n)
        x_d = ctx.enqueue_create_buffer[dtype](x_len)
        x = ctx.enqueue_create_host_buffer[dtype](x_len)
        y_d = ctx.enqueue_create_buffer[dtype](y_len)
        y = ctx.enqueue_create_host_buffer[dtype](y_len)

        generate_random_arr[dtype, m * n](A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, x_len](x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, y_len](y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)
        var beta = generate_random_scalar[dtype](-100, 100)

        # Compute norms for error checks
        var norm_A = frobenius_norm[dtype](A.unsafe_ptr(), m * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), x_len)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), y_len)

        blas_gemv[dtype](
            trans,
            m, n,
            alpha,
            A_d.unsafe_ptr(), n,
            x_d.unsafe_ptr(), 1,
            beta,
            y_d.unsafe_ptr(), 1,
            ctx,
        )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()
        for i in range(m * n):
            py_A.append(A[i])
        for i in range(x_len):
            py_x.append(x[i])
        for i in range(y_len):
            py_y.append(y[i])

        var sp_res: PythonObject
        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            sp_res = sp_blas.sgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            sp_res = sp_blas.dgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)
        else:
            print("Unsupported type: ", dtype)
            return

        # Referred to BLAS++ for an alternative error computation
        # https://github.com/icl-utk-edu/blaspp/blob/master/test/check_gemm.hh
        # NOTE: might use this for dot, gemv, ger, geru, gemm, symv, hemv, symm, trmv, trsv?, trmm, trsm?
        with y_d.map_to_host() as res_mojo:
            # Compute norm of (y - y_ref) vector
            var norm_diff = Scalar[dtype](0)
            for i in range(y_len):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])
                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)
            # From BLAS++: treat y as 1 x Ym matrix with ld = incy; k = Xm is reduction dimension
            var ok = check_gemm_error[dtype](1, y_len, x_len, alpha, beta, norm_A, norm_x, norm_y, norm_diff)
            assert_true(ok)


def ger_test[
    dtype: DType,
    m:  Int,
    n: Int,
]():
    with DeviceContext() as ctx:
        A_device = ctx.enqueue_create_buffer[dtype](m*n)
        A = ctx.enqueue_create_host_buffer[dtype](m*n)
        x_device = ctx.enqueue_create_buffer[dtype](m)
        x = ctx.enqueue_create_host_buffer[dtype](m)
        y_device = ctx.enqueue_create_buffer[dtype](n)
        y = ctx.enqueue_create_host_buffer[dtype](n)

        # Generate three arrays of random numbers on CPU
        generate_random_arr[dtype, m*n](A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, m](x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, n](y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_device, A)
        ctx.enqueue_copy(x_device, x)
        ctx.enqueue_copy(y_device, y)

        var alpha = generate_random_scalar[dtype](0.0, 1.0)

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move a and b to a SciPy-compatible array and run SciPy BLAS routine
        py_a = Python.list()
        py_x = Python.list()
        py_y = Python.list()

        for i in range(m*n):
            py_a.append(A[i])
        for i in range(m):
            py_x.append(x[i])
        for i in range(n):
            py_y.append(y[i])

        var sp_res: PythonObject
        # ger - float32
        if dtype == DType.float32:
            np_a = np.array(py_a, dtype=np.float32).reshape(m,n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            sp_res = sp_blas.sger(alpha, np_x, np_y, 1, 1, np_a)
        elif dtype == DType.float64:
            np_a = np.array(py_a, dtype=np.float64).reshape(m,n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            sp_res = sp_blas.dger(alpha, np_x, np_y, 1, 1, np_a)
        else:
            print("Unsupported type: ", dtype)
            return

        blas_ger[dtype](
            m,
            n,
            Scalar[dtype](alpha),
            x_device.unsafe_ptr(), 1,
            y_device.unsafe_ptr(), 1,
            A_device.unsafe_ptr(), n,
            ctx)

        with A_device.map_to_host() as res_mojo:
            for i in range(m):
                for j in range(n):
                    assert_almost_equal(Scalar[dtype](py=sp_res[i][j]), res_mojo[(i*n)+j], atol=atol)

                    
def syr_test[
    dtype: DType,
    n: Int,
    uplo: Int,
]():
    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](n * n)
        A = ctx.enqueue_create_host_buffer[dtype](n * n)
        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype, n * n](A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, n](x.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)

        blas_syr[dtype](uplo, n, alpha, x_d.unsafe_ptr(), 1, A_d.unsafe_ptr(), n, ctx)
        
        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas
        
        py_A = Python.list()
        py_x = Python.list()
        for i in range(n * n):
            py_A.append(A[i])
        for i in range(n):
            py_x.append(x[i])

        var sp_res: PythonObject
        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float32)
            sp_res = sp_blas.ssyr(alpha, np_x, lower=uplo, a=np_A, overwrite_a=False)
        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float64)
            sp_res = sp_blas.dsyr(alpha, np_x, lower=uplo, a=np_A, overwrite_a=False)
        else:
            print("Unsupported type: ", dtype)
            return
           
        # NOTE: Error('only 0-dimensional arrays can be converted to Python scalars')
        sp_flat = sp_res.flatten()
        with A_d.map_to_host() as res_mojo:
            for i in range(n * n):
                assert_almost_equal(res_mojo[i], Scalar[dtype](py=sp_flat[i]), atol=atol)

def syr2_test[
    dtype: DType,
    n: Int,
    uplo: Int,
]():
    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](n * n)
        A = ctx.enqueue_create_host_buffer[dtype](n * n)

        x_d = ctx.enqueue_create_buffer[dtype](n)
        x = ctx.enqueue_create_host_buffer[dtype](n)

        y_d = ctx.enqueue_create_buffer[dtype](n)
        y = ctx.enqueue_create_host_buffer[dtype](n)

        generate_random_arr[dtype, n * n](A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, n](x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, n](y.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)

        var norm_A = frobenius_norm_symmetric[dtype](A.unsafe_ptr(), n, n, uplo)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), n)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), n)

        blas_syr2[dtype](
            uplo,
            n,
            alpha,
            x_d.unsafe_ptr(), 1,
            y_d.unsafe_ptr(), 1,
            A_d.unsafe_ptr(),
            n,
            ctx,
        )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()

        for i in range(n * n):
            py_A.append(A[i])
        for i in range(n):
            py_x.append(x[i])
            py_y.append(y[i])

        var sp_res: PythonObject
        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)
            sp_res = sp_blas.ssyr2(alpha, np_x, np_y, lower=uplo, a=np_A, overwrite_a=False)

        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(n, n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)
            sp_res = sp_blas.dsyr2(alpha, np_x, np_y, lower=uplo, a=np_A, overwrite_a=False)
        else:
            print("Unsupported type: ", dtype)
            return

        sp_flat = sp_res.flatten()

        with A_d.map_to_host() as res_mojo:
            var error = InlineArray[Scalar[dtype], n*n](fill=Scalar[dtype](0))
            for i in range(n * n):
                error[i] = res_mojo[i] - Scalar[dtype](py=sp_flat[i])

            var error_norm = frobenius_norm_symmetric[dtype](
                error.unsafe_ptr(),
                n,
                n,
                uplo
            )

            var passed = check_syr_error[dtype](
                n,
                alpha,
                norm_x,
                norm_y,
                norm_A,
                error_norm
            )

            assert_true(passed)

def gbmv_test[
    dtype: DType,
    m: Int,
    n: Int,
    kl: Int,
    ku: Int,
    trans: Bool,
]():
    comptime x_len = n if not trans else m
    comptime y_len = m if not trans else n
    comptime lda = kl + ku + 1

    with DeviceContext() as ctx:
        # Dense host matrix (for reference + band extraction)
        A_dense = ctx.enqueue_create_host_buffer[dtype](m * n)

        # Band storage
        A_band = ctx.enqueue_create_host_buffer[dtype](lda * m)
        A_d = ctx.enqueue_create_buffer[dtype](lda * m)

        x = ctx.enqueue_create_host_buffer[dtype](x_len)
        x_d = ctx.enqueue_create_buffer[dtype](x_len)

        y = ctx.enqueue_create_host_buffer[dtype](y_len)
        y_d = ctx.enqueue_create_buffer[dtype](y_len)

        generate_random_arr[dtype, m * n](A_dense.unsafe_ptr(), -1, 1)
        generate_random_arr[dtype, x_len](x.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype, y_len](y.unsafe_ptr(), -100, 100)

        dense_to_band(A_dense.unsafe_ptr(), A_band.unsafe_ptr(), m, n, kl, ku)

        ctx.enqueue_copy(A_d, A_band)
        ctx.enqueue_copy(x_d, x)
        ctx.enqueue_copy(y_d, y)
        ctx.synchronize()

        var alpha = generate_random_scalar[dtype](-100, 100)
        var beta  = generate_random_scalar[dtype](-100, 100)

        var norm_A = frobenius_norm[dtype](A_dense.unsafe_ptr(), m * n)
        var norm_x = frobenius_norm[dtype](x.unsafe_ptr(), x_len)
        var norm_y = frobenius_norm[dtype](y.unsafe_ptr(), y_len)

        blas_gbmv[dtype](
            trans,
            m, n,
            kl, ku,
            alpha,
            A_d.unsafe_ptr(), lda,
            x_d.unsafe_ptr(), 1,
            beta,
            y_d.unsafe_ptr(), 1,
            ctx,
        )

        # Python reference
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_x = Python.list()
        py_y = Python.list()

        for i in range(m * n):
            py_A.append(A_dense[i])
        for i in range(x_len):
            py_x.append(x[i])
        for i in range(y_len):
            py_y.append(y[i])

        var sp_res: PythonObject

        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float32)
            np_y = np.array(py_y, dtype=np.float32)

            sp_res = sp_blas.sgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)

        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(m, n)
            np_x = np.array(py_x, dtype=np.float64)
            np_y = np.array(py_y, dtype=np.float64)

            sp_res = sp_blas.dgemv(alpha, np_A, np_x, beta=beta, y=np_y, trans=1 if trans else 0)
        else:
            print("Unsupported type")
            return

        with y_d.map_to_host() as res_mojo:
            var norm_diff = Scalar[dtype](0)
            for i in range(y_len):
                var diff = res_mojo[i] - Scalar[dtype](py=sp_res[i])

                norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)

            var ok = check_gemm_error[dtype](
                1, y_len, x_len,
                alpha, beta,
                norm_A, norm_x, norm_y,
                norm_diff
            )
            assert_true(ok)

def test_gemv():
    gemv_test[DType.float32,  64,  64, False]()
    gemv_test[DType.float32,  64,  64, True]()
    gemv_test[DType.float64,  64,  64, False]()
    gemv_test[DType.float64,  64,  64, True]()
    gemv_test[DType.float32, 1024,  64, False]()
    gemv_test[DType.float32, 1024,  64, True]()
    gemv_test[DType.float64, 1024,  64, False]()
    gemv_test[DType.float64, 1024,  64, True]()

def test_ger():
    ger_test[DType.float32, 64, 64]()
    ger_test[DType.float32, 256, 256]()
    ger_test[DType.float64, 64, 64]()
    ger_test[DType.float64, 256, 256]()
   
def test_syr():
    syr_test[DType.float32,  256, 1]()
    syr_test[DType.float32, 1024, 0]()
    syr_test[DType.float64,  256, 0]()
    syr_test[DType.float64, 1024, 1]()

def test_syr2():
    syr2_test[DType.float32,  512, 1]()
    syr2_test[DType.float32, 512, 0]()
    syr2_test[DType.float64,  512, 0]()
    syr2_test[DType.float64, 512, 1]()

def test_gbmv():
    gbmv_test[DType.float32,  64,  64, 1, 2, False]()
    gbmv_test[DType.float32,  64,  64, 2, 2, True]()
    gbmv_test[DType.float64,  64,  64, 1, 2, False]()
    gbmv_test[DType.float64,  64,  64, 2, 2, True]()
    gbmv_test[DType.float32, 512,  64, 1, 2, False]()
    gbmv_test[DType.float32, 512,  64, 2, 2, True]()
    gbmv_test[DType.float64, 512,  64, 1, 2, False]()
    gbmv_test[DType.float64, 512,  64, 2, 2, True]()

def main():
    print("--- MojoBLAS Level 2 routines testing ---")
    TestSuite.discover_tests[__functions_in_module()]().run()
