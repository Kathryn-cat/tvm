import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import relay
from tvm.meta_schedule.testing import relay_workload
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _


# m, n, p should be multiples of 16
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    A = T.match_buffer(a, (m, n), "float16")
    B = T.match_buffer(b, (n, p), "float16")
    C = T.match_buffer(c, (m, p), "float16")
    for i, j, k in T.grid(m, n, p):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


"""
example for dynamic shape:

a, b, c, m, n, p = matmul.params
func1 = matmul.specialize({a: tvm.tir.decl_buffer((128, 256))})
func2 = matmul.specialize({n: 256})
x = tvm.tir.Var("x", "int32")
func3 = matmul.specialize(
    {
        m: x * 8,
        n: x * 4,
        p: x * 2,
    }
)
print(func1)
print(func2)
print(func3)
"""

"""
a minimal tuning example:

target = tvm.target.Target("nvidia/geforce-rtx-3090")
a, b, c, m, n, p = matmul.params
with ms.Profiler() as profiler:
    sch: tvm.tir.Schedule = ms.tune_tir(
        mod=matmul.specialize({m: 256, n: 256, p: 256}),
        target=target,
        config=ms.TuneConfig(
            num_trials_per_iter=32,
            max_trials_per_task=1000,
            max_trials_global=1000,
        ),
        sch_rules=ms.default_config._DefaultCUDATensorCore.schedule_rules,
        postprocs=ms.default_config._DefaultCUDATensorCore.postprocs,
        work_dir="logs",
    )
"""
