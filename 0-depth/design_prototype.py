import argparse

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
def matmul(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (m * 128, n * 32), "float16")
    B = T.match_buffer(b, (n * 32, p * 128), "float16")
    C = T.match_buffer(c, (m * 128, p * 128), "float16")

    for i, j, k in T.grid(m * 128, p * 128, n * 32):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def apply_sch(sch):
    C = sch.get_block("C")
    i, j, k = sch.get_loops(C)
    i0, i1 = sch.split(i, [None, 128])
    j0, j1 = sch.split(j, [None, 128])
    sch.reorder(i0, j0, i1, j1, k)
    sch.blockize(i1)
    k0, k1 = sch.split(k, [None, 32])
    sch.reorder(k0, i1, j1, k1)
    CTA = sch.blockize(i1)

    i2, i3 = sch.split(i1, [None, 16])
    j2, j3 = sch.split(j1, [None, 16])
    k2, k3 = sch.split(k1, [None, 16])
    sch.reorder(i2, j2, k2, i3, j3, k3)
    C = sch.blockize(i3)

    i4, i5 = sch.split(i2, [2, None])
    j4, j5 = sch.split(j2, [2, None])
    sch.reorder(j4, i5)
    i6 = sch.fuse(i4, j4)

    sch.bind(loop=i0, thread_axis="blockIdx.y")
    sch.bind(loop=j0, thread_axis="blockIdx.x")
    sch.bind(loop=i6, thread_axis="threadIdx.y")

    A_shared = sch.cache_read(C, 1, "shared")
    B_shared = sch.cache_read(C, 2, "shared")
    C_shared = sch.cache_write(C, 0, "shared")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm", action="store_true")
    parser.add_argument("-m", type=int, default=1)
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("-p", type=int, default=1)
    args = parser.parse_args()

    target = "llvm" if args.llvm else "cuda"
    sch = tir.Schedule(matmul)
    apply_sch(sch)
    matmul_mod = tvm.build(sch.mod, target=target)
    print("Built successfully")
    sch.mod.show()

    m, n, p = args.m, args.n, args.p
    dev = tvm.cpu() if args.llvm else tvm.cuda(0)
    A_np = np.random.uniform(size=(128 * m, 32 * n)).astype("float16")
    B_np = np.random.uniform(size=(32 * n, 128 * p)).astype("float16")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((128 * m, 128 * p), dtype="float16"), dev)
    # matmul_mod(A_nd, B_nd, C_nd, m, n, p)
    evaluator = matmul_mod.time_evaluator("matmul", dev, number=10)
    time = evaluator(A_nd, B_nd, C_nd, m, n, p).mean  # in seconds
    print(time)
