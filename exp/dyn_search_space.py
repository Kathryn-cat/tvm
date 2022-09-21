import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _


@T.prim_func
def vectoradd_const(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "vectoradd_const", "tir.noalias": True})
    A = T.match_buffer(a, (256, ), "float32")
    B = T.match_buffer(b, (256, ), "float32")
    C = T.match_buffer(c, (256, ), "float32")
    for i in T.grid(256):
        with T.block("update"):
            vi = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]


@T.prim_func
def vectoradd(a: T.handle, b: T.handle, c: T.handle) -> None:
    m = T.var("int32")
    A = T.match_buffer(a, (m, ), "float32")
    B = T.match_buffer(b, (m, ), "float32")
    C = T.match_buffer(c, (m, ), "float32")
    for i in T.grid(m):
        with T.block("update"):
            vi = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]


# m, n, p should be multiples of 16
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
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


def search_space_vectoradd_const(sch: tir.Schedule) -> None:
    # don't need to cache anything?
    # vectorize / unroll?
    block_u = sch.get_block("update")
    i, = sch.get_loops(block=block_u)
    i0, i1 = sch.split(i, [None, 64])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")

def search_space_vectoradd(sch: tir.Schedule) -> None:
    # question: should we bind block/thread to var?
    block_u = sch.get_block("update")
    i, = sch.get_loops(block=block_u)
    i0, i1 = sch.split(i, [None, 2])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")

def search_space_matmul(sch: tir.Schedule) -> None:
    pass
