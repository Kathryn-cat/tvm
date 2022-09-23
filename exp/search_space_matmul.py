import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _


# m, n, p should be multiples of 16
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
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


def search_space_matmul(sch: tir.Schedule) -> None:
    # TODO: figure out how to do blockize
    # loop split and binding
    block_u = sch.get_block("update")
    i, j, k = sch.get_loops(block=block_u)
    i0, i1 = sch.split(loop=i, factors=[None, 16])
    j0, j1 = sch.split(loop=j, factors=[None, 16])
    k0, k1 = sch.split(loop=k, factors=[None, 16])
    # sch.reorder(i0, j0, k0, i1, j1, k1)
    # sch.blockize(i1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2, k2)
    #sch.unroll(k1)
    #sch.bind(i0, "blockIdx.y")
    #sch.bind(j0, "blockIdx.x")
    #sch.bind(i1, "threadIdx.y")
    #sch.bind(j1, "threadIdx.x")
    # cache read
    # A_shared = sch.cache_read(block=block_u, read_buffer_index=0, storage_scope="shared")
    # B_shared = sch.cache_read(block=block_u, read_buffer_index=1, storage_scope="shared")
    # sch.compute_at(block=A_shared, loop=k0)
    # sch.compute_at(block=B_shared, loop=k0)
    # cache write
    # C_shared = sch.cache_write(block_u, 0, "shared")
    # sch.reverse_compute_at(C_shared, j1)
    # decompose
    # sch.decompose_reduction(block_u, k0)
    # tensorization
    sch.blockize(i2)
