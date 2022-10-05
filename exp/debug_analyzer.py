import tvm
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def elementwise_symbolic(a: T.handle, b: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (128 * n, 128 * n))
    B = T.match_buffer(b, (128 * n, 128 * n))
    for i, j in T.grid(128 * n, 128 * n):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (4096 * m, 4096 * n), "float16")
    B = T.match_buffer(b, (4096 * n, 4096 * p), "float16")
    C = T.match_buffer(c, (4096 * m, 4096 * p), "float16")
    for i, j, k in T.grid(4096 * m, 4096 * n, 4096 * p):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# reverse_compute_at
def test_symbolic_1():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    _, j = sch.get_loops(block_b)
    sch.split(j, factors=[16, None])
    i1, i2, i3 = sch.get_loops(block_b)
    sch.bind(loop=i1, thread_axis="blockIdx.y")
    sch.bind(loop=i2, thread_axis="blockIdx.x")
    sch.bind(loop=i3, thread_axis="threadIdx.y")
    block_shared = sch.cache_write(block_b, 0, "shared")
    # if remove this line, no T.where()
    sch.reverse_compute_at(block_shared, i2)
    sch.mod.show()


# compute_at
def test_symbolic_2():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    _, j = sch.get_loops(block_b)
    sch.split(j, factors=[16, None])
    i1, i2, i3 = sch.get_loops(block_b)
    sch.bind(loop=i1, thread_axis="blockIdx.y")
    sch.bind(loop=i2, thread_axis="blockIdx.x")
    sch.bind(loop=i3, thread_axis="threadIdx.y")
    block_shared = sch.cache_read(block_b, 0, "shared")
    # if remove this line, no T.where()
    sch.compute_at(block_shared, i3)
    sch.mod.show()


# fusion
def test_symbolic_3():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.split(i, factors=[8, None])
    sch.split(j, factors=[16, None])
    _, i2, i3, _ = sch.get_loops(block_b)
    sch.reorder(i3, i2)
    i1, i2, _, _ = sch.get_loops(block_b)
    sch.fuse(i1, i2)
    _, i3, i4 = sch.get_loops(block_b)
    # can't fuse the line below
    sch.fuse(i3, i4)
    sch.mod.show()


# fusion - this is good
def test_symbolic_4():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.split(i, factors=[8, None])
    sch.split(j, factors=[16, None])
    i1, _, i3, _ = sch.get_loops(block_b)
    sch.reorder(i3, i1)
    i1, i2, _, _ = sch.get_loops(block_b)
    sch.fuse(i1, i2)
    _, i3, i4 = sch.get_loops(block_b)
    sch.fuse(i3, i4)
    sch.mod.show()


# decompose_reduction - this is ok
def test_symbolic_5():
    sch = tir.Schedule(matmul, debug_mask="all")
    block_c = sch.get_block("C")
    i, j, k = sch.get_loops(block_c)
    sch.decompose_reduction(block_c, k)
    sch.mod.show()


# blockize
def test_symbolic_6():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.split(i, factors=[8, None])
    sch.split(j, factors=[16, None])
    i1, i2, i3, i4 = sch.get_loops(block_b)
    # can't blockize after reordering
    sch.reorder(i2, i4, i1, i3)
    # sch.reorder(i1, i3, i2, i4)  # ok
    # sch.reorder(i3, i2, i1, i4)  # bad
    i1, i2, i3, i4 = sch.get_loops(block_b)
    sch.blockize(i3)
    sch.mod.show()


def var_dom(iters):
    """Get domains of iterators"""
    return {var: tvm.ir.Range(0, ext) for var, ext in iters}


def convert_division(divisions):
    if divisions is None or len(divisions) == 0:
        return []
    res = []
    for division in divisions[:-1]:
        res.append(
            [
                tvm.arith.normalize_iter_map_to_expr(division[0].source),
                tvm.arith.normalize_iter_map_to_expr(division[1].source),
            ]
        )
    res.append([divisions[-1][0].extent, divisions[-1][1].extent])
    return res


def test_subspace_division():
    x = tir.Var("x", "int32")
    y = tir.Var("y", "int32")
    z = tir.Var("z", "int32")
    c = tir.SizeVar("c", "int32")

    # simple 1.1
    res = tvm.arith.subspace_divide(
        [z * 12 + y * 3 + x + c], var_dom([(x, 3), (y, 4), (z, 5)]), [x]
    )
    res = convert_division(res)
    assert len(res) == 2
    tvm.ir.assert_structural_equal(res[0][0], z * 4 + y)
    tvm.ir.assert_structural_equal(res[0][1], x + c)


def test_predicate1():
    pass


if __name__ == "__main__":
    test_symbolic_6()
