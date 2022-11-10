"""
In this file, I start with handcrafting the intermediate states 
and bridging the gap between static matmul and fully dynamic 
matmul. This file focuses on the case where only the reduction
loop contains dynamic variable. The method we use is the microkernel
idea. The goal here is to align performance with the static case. 

Test with this branch with no hacks. (previous hacks in dyn-shape)
"""


import numpy as np
import torch

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.postproc import Postproc
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _
from tvm.tir.tensor_intrin.cuda import get_wmma_store_intrin

"""
Suppose we have a high-performance 128 * 128 * 32 microkernel. This will be
acquired later from MetaSchedule tuning.
Question: how to get this from MS tuning since it is not a complete IR module?
Question: is microkernel essentially register-level computation and caching?
"""


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, n: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (1024, 512 * n), "float16")
    B = T.match_buffer(b, (512 * n, 1024), "float16")
    C = T.match_buffer(c, (1024, 1024), "float16")
    for i, j, k in T.grid(1024, 1024, 512 * n):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def schedule_matmul(sch: tir.Schedule) -> None:
    # first step: blockize the subregion
    C = sch.get_block("C")
    i, j, k = sch.get_loops(C)
    i0, i1 = sch.split(i, [None, 128])
    j0, j1 = sch.split(j, [None, 128])
    k0, k1 = sch.split(k, [None, 32])
    sch.reorder(i0, j0, k0, i1, j1, k1)
    CTA = sch.blockize(i1)

    # schedule the things inside microkernel
    i2, i3 = sch.split(i1, [None, 16])
    j2, j3 = sch.split(j1, [None, 16])
    k2, k3 = sch.split(k1, [None, 16])
    sch.reorder(i2, j2, k2, i3, j3, k3)
    C = sch.blockize(i3)
    # sch.tensorize(block_or_loop=i3, tensor_intrin="wmma_sync_16x16x16_f16f16f16")

    # cache read / write
    A_wmma = sch.cache_read(C, 1, "wmma.matrix_a")
    B_wmma = sch.cache_read(C, 2, "wmma.matrix_b")
    C_wmma = sch.cache_write(C, 0, "wmma.accumulator")

    # apply tensor intrinsics
    b = sch.get_block("A_wmma.matrix_a")
    l1, l2 = sch.get_loops(b)
    l11, l12 = sch.split(l1, [None, 16])
    l21, l22 = sch.split(l2, [None, 16])
    sch.reorder(l11, l21, l12, l22)
    # sch.tensorize(block_or_loop=l12, tensor_intrin="wmma_load_16x16x16_f16_a")
    sch.blockize(loop=l12)

    b = sch.get_block("B_wmma.matrix_b")
    l1, l2 = sch.get_loops(b)
    l11, l12 = sch.split(l1, [None, 16])
    l21, l22 = sch.split(l2, [None, 16])
    sch.reorder(l11, l21, l12, l22)
    # sch.tensorize(block_or_loop=l12, tensor_intrin="wmma_load_16x16x16_f16_b")
    sch.blockize(loop=l12)

    b = sch.get_block("C_wmma.accumulator")
    l1, l2 = sch.get_loops(b)
    l11, l12 = sch.split(l1, [None, 16])
    l21, l22 = sch.split(l2, [None, 16])
    sch.reorder(l11, l21, l12, l22)
    # sch.tensorize(block_or_loop=l12, tensor_intrin="wmma_store_16x16x16_f16_shared")
    sch.blockize(loop=l12)


def test(build=False):
    sch = tir.Schedule(matmul, debug_mask="all")
    schedule_matmul(sch)
    sch.mod.show()

    if build:
        matmul_mod = tvm.build(sch.mod, target="llvm")

        # testing
        # dev = tvm.cuda(0)
        dev = tvm.cpu()
        A_np = np.random.uniform(size=(1024, 512)).astype("float16")
        B_np = np.random.uniform(size=(512, 1024)).astype("float16")
        A_nd = tvm.nd.array(A_np, dev)
        B_nd = tvm.nd.array(B_np, dev)
        C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float16"), dev)
        # calculate numpy multiplication results
        device = torch.device("cuda:0")
        C_np = torch.tensor(A_np).to(device) @ torch.tensor(B_np).to(device)
        # calculate tvm multiplication results
        matmul_mod(A_nd, B_nd, C_nd, 1)
        # check correctness
        np.testing.assert_allclose(C_np.detach().cpu().numpy(), C_nd.numpy(), atol=1e-3)


if __name__ == "__main__":
    test()
