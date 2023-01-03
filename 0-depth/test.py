"""
Build and debug the prototype.
"""


import numpy as np
import torch
from prototype import Module

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.postproc import Postproc
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _
from tvm.tir.tensor_intrin.cuda import get_wmma_store_intrin

if __name__ == "__main__":
    mod = Module
    mod.show()
    matmul_mod = tvm.build(mod, target="cuda")
