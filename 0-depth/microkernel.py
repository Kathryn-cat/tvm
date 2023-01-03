"""
In this file, we aim to construct a microkernel for one threadblock which has good-enough performance
per CUTLASS case.
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

# we don't have direct performance measurement for one threadblock, so we tune a huge one first.


@T.prim_func
def hgemm_4096(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "microkernel", "tir.noalias": True})
    A = T.match_buffer(a, (4096, 4096), "float16")
    B = T.match_buffer(b, (4096, 4096), "float16")
    C = T.match_buffer(c, (4096, 4096), "float16")
    for i, j, k in T.grid(4096, 4096, 4096):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_128_128_32(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "microkernel", "tir.noalias": True})
    A = T.match_buffer(a, (128, 32), "float16")
    B = T.match_buffer(b, (32, 128), "float16")
    C = T.match_buffer(c, (128, 128), "float16")
    for i, j, k in T.grid(128, 128, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def tuning(mod):
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.tune_tir(
            mod=mod,
            target=target,
            num_trials_per_iter=32,
            max_trials_global=500,
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
    sch.mod.show()
    return sch.mod


if __name__ == "__main__":
    tuning(hgemm_4096)
