import numpy as np
import torch

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.postproc import Postproc
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _
from tvm.tir.tensor_intrin.cuda import get_wmma_store_intrin


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, n: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (16 * n, 128), "float16")
    B = T.match_buffer(b, (128, 128), "float16")
    C = T.match_buffer(c, (16 * n, 128), "float16")
    for i, j, k in T.grid(16 * n, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def schedule_matmul(sch):
    b = sch.get_block("C")
    i, j, k = sch.get_loops(b)
    i0, i1 = sch.split(i, [None, 16])
    j0, j1 = sch.split(j, [None, 16])
    k0, k1 = sch.split(k, [None, 16])
    sch.reorder(i0, j0, k0, i1, j1, k1)
    CTA = sch.blockize(i1)
    b = sch.cache_write(block=CTA, write_buffer_index=0, storage_scope="shared")
    sch.reverse_compute_at(block=b, loop=j0, preserve_unit_loops=True, index=-1)
    sch.mod.show()


if __name__ == "__main__":
    sch = tir.Schedule(matmul, debug_mask="all")
    schedule_matmul(sch)
