"""
Build and debug the prototype.
"""


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_pad", action="store_true")
    args = parser.parse_args()

    if args.no_pad:
        from prototype_no_pad import Module
    else:
        from prototype import Module
    mod = Module
    mod.show()
    """
    matmul_mod = tvm.build(mod, target="cuda")
    print("Built successfully")
    dev = tvm.cuda(0)

    m, n, p = 128, 32, 128
    A_np = np.random.uniform(size=(m, n)).astype("float16")
    B_np = np.random.uniform(size=(n, p)).astype("float16")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((m, p), dtype="float16"), dev)
    matmul_mod(A_nd, B_nd, C_nd)
    # evaluator = matmul_mod.time_evaluator("StagedModule", dev, number=10)
    # time = evaluator(A_nd, B_nd, C_nd).mean
    # print(time)
    """
