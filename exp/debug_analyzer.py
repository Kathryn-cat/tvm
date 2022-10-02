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


def test_split_symbolic():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    _, j = sch.get_loops(block_b)
    sch.split(j, factors=[16, None])
    print(sch.mod.script())


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
    test_split_symbolic()
    test_subspace_division()
