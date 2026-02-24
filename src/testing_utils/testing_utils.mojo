from random import rand, seed

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

