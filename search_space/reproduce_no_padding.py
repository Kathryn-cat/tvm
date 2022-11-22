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


# reproduce DietCode experiment 1
# in this file, input is guaranteed to be multiples of microkernel, so no padding is needed
# assumption is that n is a multiple of 16, so n * 16 = k * 256
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


# specify the shape of microkernel in scheduling
def schedule_matmul(sch, d1_val, d2_val, d3_val):
    C = sch.get_block("C")
    i, j, k = sch.get_loops(C)
    i0, i1 = sch.split(i, [None, d1_val])
    j0, j1 = sch.split(j, [None, d2_val])
    k0, k1 = sch.split(k, [None, d3_val])
    sch.reorder(i0, j0, k0, i1, j1, k1)

    sch_128_128_32_part1(sch)


def test(mod, d1_val, d2_val, d3_val, k=1, build=False):
    # _, _, _, _, d1, d2, d3 = mod.params
    # mod = mod.specialize({d1: d1_val, d2: d2_val, d3: d3_val})
    sch = tir.Schedule(mod, debug_mask="all")
    schedule_matmul(sch, d1_val, d2_val, d3_val)
    sch.mod.show()

    if build:
        matmul_mod = tvm.build(sch.mod, target="llvm")

        # testing
        # dev = tvm.cuda(0)
        dev = tvm.cpu()
        A_np = np.random.uniform(size=(256 * k, 768)).astype("float16")
        B_np = np.random.uniform(size=(768, 2304)).astype("float16")
        A_nd = tvm.nd.array(A_np, dev)
        B_nd = tvm.nd.array(B_np, dev)
        C_nd = tvm.nd.array(np.zeros((256 * k, 2304), dtype="float16"), dev)
        # calculate numpy multiplication results
        device = torch.device("cuda:0")
        C_np = torch.tensor(A_np).to(device) @ torch.tensor(B_np).to(device)
        # calculate tvm multiplication results
        # matmul_mod(A_nd, B_nd, C_nd, k)
        # check correctness
        # np.testing.assert_allclose(C_np.detach().cpu().numpy(), C_nd.numpy(), atol=2.0)

        """
        # measure running time
        num_flop = 2 * 128 * 768 * 2304
        evaluator = matmul_mod.time_evaluator("matmul", dev, number=10)
        print("matmul: %f GFLOPS\n" % (num_flop / evaluator(A_nd, B_nd, C_nd, 1).mean / 1e9))
        """


if __name__ == "__main__":
    """
    mod, trace = microkernel_tuning(ALL_MICROKERNELS[7])
    mod.show()
    print(trace)
    """
    test(matmul1, 128, 128, 32, 1, True)
