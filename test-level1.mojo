from testing import assert_equal, TestSuite
from sys import has_accelerator
from gpu.host import DeviceContext
from gpu import block_dim, grid_dim, thread_idx

from src import iamax_device
from random import rand, seed
from math import ceildiv
from python import Python

comptime dtype = DType.float32
comptime size = 51
comptime TBsize = 512

def test_iamax():
    with DeviceContext() as ctx:
        print("iamax test")

        # Allocate GPU and CPU memory
        d_v = ctx.enqueue_create_buffer[dtype](size)
        v = ctx.enqueue_create_host_buffer[dtype](size)

        # Generate an array of random numbers on CPU
        seed()
        rand[dtype](v.unsafe_ptr(), size)

        # Copy random vector from CPU to GPU memory
        ctx.enqueue_copy(d_v, v)

        # Allocate memory for a single int on GPU to store result, initialize to -1
        d_res = ctx.enqueue_create_buffer[DType.int64](1)
        d_res.enqueue_fill(-1)

        # Launch Mojo GPU kernel
        comptime kernel = iamax_device[TBsize]
        ctx.enqueue_function[kernel, kernel](
            size, d_v,
            1, d_res,
            grid_dim=ceildiv(size, TBsize),     # total thread blocks
            block_dim=TBsize                    # threads per block
        )

        # Import SciPy and numpy
        sp = Python.import_module("scipy")
        np = Python.import_module("numpy")
        sp_blas = sp.linalg.blas

        # Move values in v to a SciPy-compatible array
        py_list = Python.list()
        for i in range(size):
            py_list.append(v[i])
        np_v = np.array(py_list, dtype=np.float32)

        # Run SciPy BLAS routine
        sp_res = sp_blas.isamax(np_v)

        # Move Mojo result from CPU to GPU and compare to SciPy
        sp_res_mojo = Int(py=sp_res)             # cast Python int into Mojo int
        with d_res.map_to_host() as res_mojo:
            print("out:", res_mojo[0])
            print("expected:", sp_res)
            assert_equal(res_mojo[0], sp_res_mojo)


def main():
    print("--- MojoBLAS Level 1 routines testing ---")
    TestSuite.discover_tests[__functions_in_module()]().run()
