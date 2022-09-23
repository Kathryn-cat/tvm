import numpy as np

import tvm
from tvm.script import tir as T


@T.prim_func
def matmul1(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    m = T.var("int32")
    n = T.var("int32")
    p = T.var("int32")
    A = T.match_buffer(a, (m, n), "float16")
    B = T.match_buffer(b, (n, p), "float16")
    C = T.match_buffer(c, (m, p), "float16")
    for i, j, k in T.grid(m, n, p):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

@T.prim_func
def matmul2(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (16 * m, 16 * n), "float16")
    B = T.match_buffer(b, (16 * n, 16 * p), "float16")
    C = T.match_buffer(c, (16 * m, 16 * p), "float16")
    for i, j, k in T.grid(16 * m, 16 * n, 16 * p):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


sch = tvm.tir.Schedule(matmul1)
i, j, k = sch.get_loops("update")
i, ii = sch.split(i, factors=[None, 16])
j, ji = sch.split(j, factors=[None, 16])
k, ki = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k, ii, ji, ki)
block_mm = sch.blockize(ii)
sch.mod.show()
mm_mod = tvm.build(sch.mod, target="llvm")
