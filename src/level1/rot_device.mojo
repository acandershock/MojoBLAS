from gpu import grid_dim, block_dim, block_idx, thread_idx

# level1.rot
# applies a plane rotation to vectors x and y
fn rot_device[
    BLOCK: Int,
    dtype: DType
](
    n: Int,
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incx: Int,
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    incy: Int,
    c: Scalar[dtype],
    s: Scalar[dtype]
):
    if (n < 1):
        return

    var global_tid = block_idx.x * block_dim.x + thread_idx.x
    var n_threads = grid_dim.x * block_dim.x

    # Each thread processes multiple elements with stride
    for i in range(global_tid, n, n_threads):
        var ix = i * incx
        var iy = i * incy

        var tmp = c * x[ix] + s * y[iy]
        y[iy] = c * y[iy] - s * x[ix]
        x[ix] = tmp
