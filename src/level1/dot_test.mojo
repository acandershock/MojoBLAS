from math import ceildiv
from buffer import NDBuffer, DimList
from algorithm import sum
from gpu import thread_idx, block_idx, block_dim, barrier, lane_id
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.warp import sum as warp_sum, WARP_SIZE
from gpu.memory import AddressSpace
from algorithm.functional import elementwise
from layout import Layout, LayoutTensor
from utils import IndexList
from testing import assert_equal
from random import random_float64
from dot import dot_product

alias dtype = DType.float16


fn test[test_size: Int]() raises:
    var a = UnsafePointer[Scalar[dtype]].alloc(test_size)
    var b = UnsafePointer[Scalar[dtype]].alloc(test_size)
    for i in range(test_size):
        a[i] = i+1
        b[i] = i+1

    var s = dot_product[dtype, test_size](a, b)

    var expected_sum: Scalar[dtype] = 0
    for i in range(test_size):
        expected_sum += a[i] * b[i]

    assert_equal(expected_sum, s)
    print("Test for ", test_size, " passes")

def main():
    test[1]()
    test[3]()
    test[31]()
    test[32]()
    test[64]()
    test[128]()
    test[129]()
    test[512]()
    test[1024]()
    test[65536]()
