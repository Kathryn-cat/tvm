"""
In this file, I start with handcrafting the intermediate states 
and bridging the gap between static matmul and fully dynamic 
matmul. This file focuses on the case where only the reduction
loop contains dynamic variable. The method we use is the microkernel
idea. The goal here is to align performance with the static case. 
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
def matmul(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (1024, 512 * n), "float16")
    B = T.match_buffer(b, (512 * n, 1024), "float16")
    C = T.match_buffer(c, (1024, 1024), "float16")
    for i, j, k in T.grid(1024, 512 * n, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
