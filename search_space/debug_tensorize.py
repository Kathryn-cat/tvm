import numpy as np
import torch
from microkernels import *

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.postproc import Postproc
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _
from tvm.tir.tensor_intrin.cuda import get_wmma_store_intrin


@T.prim_func
def matmul1(a: T.handle, b: T.handle, c: T.handle, k: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (256 * k, 768), "float16")
    B = T.match_buffer(b, (768, 2304), "float16")
    C = T.match_buffer(c, (256 * k, 2304), "float16")
    for i, j, k in T.grid(256 * k, 2304, 768):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


sch = tir.Schedule(matmul1, debug_mask="all")
C = sch.get_block("C")
i, j, k = sch.get_loops(C)
i0, i1 = sch.split(i, [None, 128])
j0, j1 = sch.split(j, [None, 128])
k0, k1 = sch.split(k, [None, 32])
sch.reorder(i0, j0, k0, i1, j1, k1)

b0 = sch.get_block(name="C", func_name="main")
b1 = sch.get_block(name="root", func_name="main")
b0 = sch.get_block(name="C", func_name="main")
i, j, k, l5, l6, l7 = sch.get_loops(block=b0)
v24, v25, v26, v27, v28, v = sch.sample_perfect_tile(
    loop=l5, n=6, max_innermost_factor=4, decision=[2, 4, 1, 1, 1, 16]
)
l29, l30, l31, l32, l33, l01 = sch.split(
    loop=l5, factors=[v24, v25, v26, v27, v28, v], preserve_unit_iters=True
)
v34, v35, v36, v37, v38, v = sch.sample_perfect_tile(
    loop=l6, n=6, max_innermost_factor=4, decision=[2, 2, 2, 1, 1, 16]
)
l39, l40, l41, l42, l43, l02 = sch.split(
    loop=l6, factors=[v34, v35, v36, v37, v38, v], preserve_unit_iters=True
)
v44, v45, v46, v = sch.sample_perfect_tile(
    loop=l7, n=4, max_innermost_factor=4, decision=[1, 2, 1, 16]
)
l47, l48, l49, l03 = sch.split(loop=l7, factors=[v44, v45, v46, v], preserve_unit_iters=True)
sch.reorder(l29, l30, l31, l32, l33, l39, l40, l41, l42, l43, l47, l48, l49, l01, l02, l03)
b20 = sch.blockize(loop=l01)

sch.reorder(i, j, l29, l39, l30, l40, l31, l41, k, l47, l48, l32, l42, l49, l33, l43)
sch.bind(loop=i, thread_axis="blockIdx.y")
l51 = sch.fuse(j, l29, preserve_unit_iters=True)
l51 = sch.fuse(l51, l39, preserve_unit_iters=True)  # change line 1
l51 = sch.fuse(l51, l30, preserve_unit_iters=True)
l51 = sch.fuse(l51, l40, preserve_unit_iters=True)  # change line 2
sch.bind(loop=l51, thread_axis="blockIdx.x")
l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
sch.bind(loop=l52, thread_axis="threadIdx.y")
l47 = sch.fuse(k, l47, preserve_unit_iters=True)

sch.blockize(l47)
b53 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="shared")
b54 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="wmma.accumulator")

l59, l60 = sch.get_loops(block=b54)
l61, l62 = sch.split(loop=l60, factors=[None, 16], preserve_unit_iters=True)
l63, l64 = sch.split(loop=l59, factors=[None, 16], preserve_unit_iters=True)
l68, l69, l70, l71 = sch.get_loops(block=b54)
sch.reorder(l68, l70, l69, l71)
b72 = sch.blockize(loop=l69)

# sch.tensorize(block_or_loop=b72, tensor_intrin="wmma_store_16x16x16_f16_shared")

sch.mod.show()
