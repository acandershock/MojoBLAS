from sys import argv
from testing import assert_equal, assert_almost_equal, assert_true, TestSuite
from gpu.host import DeviceContext

from src import *
from python import Python, PythonObject


def gemm_test[
    dtype: DType,
    m: Int,
    n: Int,
    k: Int,
    trans_a: Bool, trans_b: Bool
]() :
    with DeviceContext() as ctx:
        A_d = ctx.enqueue_create_buffer[dtype](m * k)
        A = ctx.enqueue_create_host_buffer[dtype](m * k)
        B_d = ctx.enqueue_create_buffer[dtype](k * n)
        B = ctx.enqueue_create_host_buffer[dtype](k * n)
        C_d = ctx.enqueue_create_buffer[dtype](m * n)
        C = ctx.enqueue_create_host_buffer[dtype](m * n)

        generate_random_arr[dtype](m * k, A.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](k * n, B.unsafe_ptr(), -100, 100)
        generate_random_arr[dtype](m * n, C.unsafe_ptr(), -100, 100)

        ctx.enqueue_copy(A_d, A)
        ctx.enqueue_copy(B_d, B)
        ctx.enqueue_copy(C_d, C)

        var alpha = generate_random_scalar[dtype](-100, 100)
        var beta = generate_random_scalar[dtype](-100, 100)

        var norm_A = frobenius_norm[dtype](A.unsafe_ptr(), m * k)
        var norm_B = frobenius_norm[dtype](B.unsafe_ptr(), k * n)
        var norm_C = frobenius_norm[dtype](C.unsafe_ptr(), m * n)

        var lda = m if trans_a else k
        var ldb = k if trans_b else n
        var ldc = n

        blas_gemm[dtype](
            trans_a, trans_b,
            m, n, k,
            alpha,
            A_d.unsafe_ptr(), lda,
            B_d.unsafe_ptr(), ldb,
            beta,
            C_d.unsafe_ptr(), ldc,
            ctx
        )

        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        py_A = Python.list()
        py_B = Python.list()
        py_C = Python.list()
        for i in range(m * k):
            py_A.append(A[i])
        for i in range(k * n):
            py_B.append(B[i])
        for i in range(m * n):
            py_C.append(C[i])

        if dtype == DType.float32:
            np_A = np.array(py_A, dtype=np.float32).reshape(m, k) if not trans_a else np.array(py_A, dtype=np.float32).reshape(k, m)
            np_B = np.array(py_B, dtype=np.float32).reshape(k, n) if not trans_b else np.array(py_B, dtype=np.float32).reshape(n, k)
            np_C = np.array(py_C, dtype=np.float32).reshape(m, n)               

        elif dtype == DType.float64:
            np_A = np.array(py_A, dtype=np.float64).reshape(m, k) if not trans_a else np.array(py_A, dtype=np.float64).reshape(k, m)
            np_B = np.array(py_B, dtype=np.float64).reshape(k, n) if not trans_b else np.array(py_B, dtype=np.float64).reshape(n, k)
            np_C = np.array(py_C, dtype=np.float64).reshape(m, n) 
            #sp_res = sp_blas.dgemm(alpha, np_A, np_B, beta=beta, c=np_C, trans_a= 0 if trans_a else 1, trans_b= 0 if trans_b else 1)
        else :
            print("Unsupported type: ", dtype)
            return
        
        var op_A = np.transpose(np_A) if trans_a else np_A
        var op_B = np.transpose(np_B) if trans_b else np_B

        var sp_res = alpha * np.matmul(op_A, op_B) + beta * np_C
        
        with C_d.map_to_host() as res_mojo :
            var norm_diff = Scalar[dtype](0)
            for i in range(m):
                for j in range(n) :
                    var diff = res_mojo[i * n + j] - Scalar[dtype](py=sp_res[i][j])
                    norm_diff += diff * diff
            norm_diff = sqrt(norm_diff)
            var ok = check_gemm_error[dtype](m, n, k, alpha, beta, norm_A, norm_B, norm_C, norm_diff)
            assert_true(ok)

def test_gemm() :

    gemm_test[DType.float32, 64, 64, 64, False, False]()
    gemm_test[DType.float32, 64, 64, 64, True, False]()
    gemm_test[DType.float32, 64, 64, 64, False, True]()
    gemm_test[DType.float32, 64, 64, 64, True, True]()

    gemm_test[DType.float32, 64, 1024, 64, False, False]()
    gemm_test[DType.float32, 64, 1024, 64, True, False]()
    gemm_test[DType.float32, 64, 1024, 64, False, True]()
    gemm_test[DType.float32, 64, 1024, 64, True, True]()

    gemm_test[DType.float32, 1024, 64, 64, False, False]()
    gemm_test[DType.float32, 1024, 64, 64, True, False]()
    gemm_test[DType.float32, 1024, 64, 64, False, True]()
    gemm_test[DType.float32, 1024, 64, 64, True, True]()

    gemm_test[DType.float32, 64, 64, 1024, False, False]()
    gemm_test[DType.float32, 64, 64, 1024, True, False]()
    gemm_test[DType.float32, 64, 64, 1024, False, True]()
    gemm_test[DType.float32, 64, 64, 1024, True, True]()

    gemm_test[DType.float32, 1024, 64, 1024, False, False]()
    gemm_test[DType.float32, 1024, 64, 1024, True, False]()
    gemm_test[DType.float32, 1024, 64, 1024, False, True]()
    gemm_test[DType.float32, 1024, 64, 1024, True, True]()

    gemm_test[DType.float32, 64, 1024, 1024, False, False]()
    gemm_test[DType.float32, 64, 1024, 1024, True, False]()
    gemm_test[DType.float32, 64, 1024, 1024, False, True]()
    gemm_test[DType.float32, 64, 1024, 1024, True, True]()

    gemm_test[DType.float32, 1024, 1024, 64, False, False]()
    gemm_test[DType.float32, 1024, 1024, 64, True, False]()
    gemm_test[DType.float32, 1024, 1024, 64, False, True]()
    gemm_test[DType.float32, 1024, 1024, 64, True, True]()

    gemm_test[DType.float32, 1024, 1024, 1024, False, False]()
    gemm_test[DType.float32, 1024, 1024, 1024, True, False]()
    gemm_test[DType.float32, 1024, 1024, 1024, False, True]()
    gemm_test[DType.float32, 1024, 1024, 1024, True, True]()

    gemm_test[DType.float64, 64, 64, 64, False, False]()
    gemm_test[DType.float64, 64, 64, 64, True, False]()
    gemm_test[DType.float64, 64, 64, 64, False, True]()
    gemm_test[DType.float64, 64, 64, 64, True, True]()

    gemm_test[DType.float64, 64, 1024, 64, False, False]()
    gemm_test[DType.float64, 64, 1024, 64, True, False]()
    gemm_test[DType.float64, 64, 1024, 64, False, True]()
    gemm_test[DType.float64, 64, 1024, 64, True, True]()

    gemm_test[DType.float64, 1024, 64, 64, False, False]()
    gemm_test[DType.float64, 1024, 64, 64, True, False]()
    gemm_test[DType.float64, 1024, 64, 64, False, True]()
    gemm_test[DType.float64, 1024, 64, 64, True, True]()

    gemm_test[DType.float64, 64, 64, 1024, False, False]()
    gemm_test[DType.float64, 64, 64, 1024, True, False]()
    gemm_test[DType.float64, 64, 64, 1024, False, True]()
    gemm_test[DType.float64, 64, 64, 1024, True, True]()

    gemm_test[DType.float64, 1024, 64, 1024, False, False]()
    gemm_test[DType.float64, 1024, 64, 1024, True, False]()
    gemm_test[DType.float64, 1024, 64, 1024, False, True]()
    gemm_test[DType.float64, 1024, 64, 1024, True, True]()

    gemm_test[DType.float64, 64, 1024, 1024, False, False]()
    gemm_test[DType.float64, 64, 1024, 1024, True, False]()
    gemm_test[DType.float64, 64, 1024, 1024, False, True]()
    gemm_test[DType.float64, 64, 1024, 1024, True, True]()

    gemm_test[DType.float64, 1024, 1024, 64, False, False]()
    gemm_test[DType.float64, 1024, 1024, 64, True, False]()
    gemm_test[DType.float64, 1024, 1024, 64, False, True]()
    gemm_test[DType.float64, 1024, 1024, 64, True, True]()

    gemm_test[DType.float64, 1024, 1024, 1024, False, False]()
    gemm_test[DType.float64, 1024, 1024, 1024, True, False]()
    gemm_test[DType.float64, 1024, 1024, 1024, False, True]()
    gemm_test[DType.float64, 1024, 1024, 1024, True, True]()


def main():
    print("--- MojoBLAS Level 2 routines testing ---")
    var args = argv()
    if (len(args) < 2):
        TestSuite.discover_tests[__functions_in_module()]().run()
        return

    var suite = TestSuite(cli_args=List[StaticString]())
    for i in range(1, len(args)):
        if   args[i] == "gemm":  suite.test[test_gemm]()
        else: print("unknown routine:", args[i])
    suite^.run()








