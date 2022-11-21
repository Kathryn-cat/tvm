import numpy as np
import torch

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.postproc import Postproc
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _
from tvm.tir.tensor_intrin.cuda import get_wmma_store_intrin

# representative microkernels from CUTLASS


@T.prim_func
def microkernel_128_128_32(
    A: T.Buffer[(128, 32), "float16"],
    B: T.Buffer[(32, 128), "float16"],
    C: T.Buffer[(128, 128), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_128_128_32", "tir.noalias": True})
    for i, j, k in T.grid(128, 128, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_256_128_32(
    A: T.Buffer[(256, 32), "float16"],
    B: T.Buffer[(32, 128), "float16"],
    C: T.Buffer[(256, 128), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_256_128_32", "tir.noalias": True})
    for i, j, k in T.grid(256, 128, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_128_256_32(
    A: T.Buffer[(128, 32), "float16"],
    B: T.Buffer[(32, 256), "float16"],
    C: T.Buffer[(128, 256), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_128_256_32", "tir.noalias": True})
    for i, j, k in T.grid(128, 256, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_64_256_32(
    A: T.Buffer[(64, 32), "float16"],
    B: T.Buffer[(32, 256), "float16"],
    C: T.Buffer[(64, 256), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_64_256_32", "tir.noalias": True})
    for i, j, k in T.grid(64, 256, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_256_64_32(
    A: T.Buffer[(256, 32), "float16"],
    B: T.Buffer[(32, 64), "float16"],
    C: T.Buffer[(256, 64), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_256_64_32", "tir.noalias": True})
    for i, j, k in T.grid(256, 64, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_64_64_32(
    A: T.Buffer[(64, 32), "float16"],
    B: T.Buffer[(32, 64), "float16"],
    C: T.Buffer[(64, 64), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_64_64_32", "tir.noalias": True})
    for i, j, k in T.grid(64, 64, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_128_64_32(
    A: T.Buffer[(128, 32), "float16"],
    B: T.Buffer[(32, 64), "float16"],
    C: T.Buffer[(128, 64), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_128_64_32", "tir.noalias": True})
    for i, j, k in T.grid(128, 64, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def microkernel_64_128_32(
    A: T.Buffer[(64, 32), "float16"],
    B: T.Buffer[(32, 128), "float16"],
    C: T.Buffer[(64, 128), "float16"],
) -> None:
    T.func_attr({"global_symbol": "microkernel_64_128_32", "tir.noalias": True})
    for i, j, k in T.grid(64, 128, 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


ALL_MICROKERNELS = [
    microkernel_128_128_32,
    microkernel_256_128_32,
    microkernel_128_256_32,
    microkernel_64_256_32,
    microkernel_256_64_32,
    microkernel_64_64_32,
    microkernel_128_64_32,
    microkernel_64_128_32,
]


def microkernel_tuning(mod):
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
        sch = tir.Schedule(mod, debug_mask="all")
        trace.apply_to_schedule(sch, False)
    return sch.mod, trace


if __name__ == "__main__":
    mod, trace = microkernel_tuning(ALL_MICROKERNELS[1])
    mod.show()
    print(trace)
