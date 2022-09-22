import numpy as np

import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer[(1024, 1024), "float32"],
        B: T.Buffer[(1024, 1024), "float32"],
        C: T.Buffer[(1024, 1024), "float32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
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


sch = tvm.tir.Schedule(matmul)
i, j, k = sch.get_loops("update")
i, ii = sch.split(i, factors=[None, 16])
j, ji = sch.split(j, factors=[None, 16])
k, ki = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k, ii, ji, ki)
block_mm = sch.blockize(ii)
sch.mod.show()
