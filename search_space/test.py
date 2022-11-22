from typing import List

import numpy as np
import torch

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T


# we handle all the shapes of multiple of 16
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    m = T.var("int32")
    n = T.var("int32")
    p = T.var("int32")
    A = T.match_buffer(a, (m, n), "float16")
    B = T.match_buffer(b, (n, p), "float16")
    C = T.match_buffer(c, (m, p), "float16")

    A_pad = T.alloc_buffer(((m + 127) // 128 * 128, (n + 31) // 32 * 32), "float16")
    B_pad = T.alloc_buffer(((n + 31) // 32 * 32, (p + 127) // 128 * 128), "float16")
    C_pad = T.alloc_buffer(((m + 127) // 128 * 128, (p + 127) // 128 * 128), "float16")
    for i, j in T.grid((m + 127) // 128 * 128, (n + 31) // 32 * 32):
        with T.block("A_pad"):
            vi, vj = T.axis.remap("SS", [i, j])
            A_pad[vi, vj] = T.if_then_else(
                vi < m and vj < n, A[vi, vj], T.float16(0), dtype="float16"
            )
    for i, j in T.grid((n + 31) // 32 * 32, (p + 127) // 128 * 128):
        with T.block("B_pad"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_pad[vi, vj] = T.if_then_else(
                vi < n and vj < p, B[vi, vj], T.float16(0), dtype="float16"
            )
    for i, j, k in T.grid((m + 127) // 128 * 128, (p + 127) // 128 * 128, (n + 31) // 32 * 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C_pad[vi, vj] = T.float16(0.0)
            C_pad[vi, vj] = C_pad[vi, vj] + A_pad[vi, vk] * B_pad[vk, vj]
    for i, j in T.grid(m, p):
        with T.block("C_pad"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = C_pad[vi, vj]


sch = tir.Schedule(matmul, debug_mask="all")
C = sch.get_block("C")
i, j, k = sch.get_loops(C)
i0, i1 = sch.split(i, [None, 128])
j0, j1 = sch.split(j, [None, 128])
sch.reorder(i0, j0, i1, j1, k)
sch.blockize(i1)
sch.mod.show()
