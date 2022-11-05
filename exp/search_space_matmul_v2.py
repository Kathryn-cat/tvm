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
