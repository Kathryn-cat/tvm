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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm", action="store_true")
    args = parser.parse_args()

    target = "llvm" if args.llvm else "cuda"
    matmul_mod = tvm.build(matmul, target=target)
    print("Built successfully")
    matmul.show()

    m, n, p = 1, 1, 1
    dev = tvm.cpu() if args.llvm else tvm.cuda(0)
    A_np = np.random.uniform(size=(128 * m, 32 * n)).astype("float16")
    B_np = np.random.uniform(size=(32 * n, 128 * p)).astype("float16")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((128 * m, 128 * p), dtype="float16"), dev)
    matmul_mod(A_nd, B_nd, C_nd, m, n, p)
