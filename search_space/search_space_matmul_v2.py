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
acquired later from MetaSchedule tuning, but now we just leave it as a
subregion computation.
"""


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, n: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (1024, 512 * n), "float32")
    B = T.match_buffer(b, (512 * n, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")
    for i, j, k in T.grid(1024, 1024, 512 * n):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def schedule_matmul(sch: tir.Schedule) -> None:
    # first step: blockize the subregion
    C = sch.get_block("C")
    i, j, k = sch.get_loops(C)


def test():
    sch = tir.Schedule(matmul, debug_mask="all")
    schedule_matmul(sch)
    matmul_mod = tvm.build(sch.mod, target="llvm")

    # testing
    # dev = tvm.cuda(0)
    dev = tvm.cpu()
    A_np = np.random.uniform(size=(1024, 512)).astype("float32")
    B_np = np.random.uniform(size=(512, 1024)).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
    # calculate numpy multiplication results
    device = torch.device("cuda:0")
    C_np = torch.tensor(A_np).to(device) @ torch.tensor(B_np).to(device)
    # calculate tvm multiplication results
    matmul_mod(A_nd, B_nd, C_nd, 1)
    # check correctness
    np.testing.assert_allclose(C_np.detach().cpu().numpy(), C_nd.numpy(), atol=1e-3)


if __name__ == "__main__":
    test()
