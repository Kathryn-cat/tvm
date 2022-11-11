"""
In this file, I start with handcrafting the intermediate states 
and bridging the gap between static matmul and fully dynamic 
matmul. This file focuses on the case where only the reduction
loop contains dynamic variable. The goal here is to align performance
with the static case. 

The method here is to saturate a thread and tile up, using a 128 * 128 * 32
microkernel, and see if the performance can be aligned with
MetaSchedule performance.

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


@T.prim_func
def microkernel(
    A: T.Buffer[(128, 32), "float16"],
    B: T.Buffer[(32, 128), "float16"],
    C: T.Buffer[(128, 128), "float16"],
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i, j, k in T.grid(128, 128, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def tune_microkernel(mod):
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.tune_tir(
            mod=mod,
            target=target,
            num_trials_per_iter=32,
            max_trials_global=1000,
            work_dir="logs",
        )
        # find the best trace
        min_run_time = 1.0
        best_idx = -1
        for i, record in enumerate(sch.get_all_tuning_records()):
            if record.run_secs[0].value < min_run_time:
                min_run_time = record.run_secs[0].value
                best_idx = i
        print(f"best tuning record is {best_idx}, min_run_time is {min_run_time}")
        trace = sch.get_all_tuning_records()[best_idx].trace
        print("trace:")
        print(trace)
        sch = tir.Schedule(mod, debug_mask="all")
        trace.apply_to_schedule(sch, False)
    return sch.mod


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


def schedule_matmul(sch):
    pass


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
    optimized_mod = tune_microkernel(microkernel)
    optimized_mod.show()
