import numpy as np

import tvm
from tvm import meta_schedule as ms
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


def default_schedule_rules():
    from tvm.meta_schedule import schedule_rule as M
    from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group

    return [
        M.MultiLevelTilingTensorCore(
            intrin_groups=[
                get_wmma_intrin_group(
                    store_scope="shared",
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    trans_b=trans_b,
                )
                for (in_dtype, out_dtype) in [("float16", "float16"), ("int8", "int32")]
                for trans_b in [False, True]
            ],
            structure="SSSRRSRS",
            tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y"],
            max_innermost_factor=4,
            vector_load_lens=[1, 2, 3, 4, 8, 16],
            reuse_read=M.ReuseType(req="must", levels=[4], scope="shared"),
            reuse_write=M.ReuseType(
                req="must",
                levels=[2],
                scope="shared",
            ),
            use_software_pipeline=False,
        ),
        M.MultiLevelTiling(
            structure="SSSRRSRS",
            tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
            max_innermost_factor=64,
            vector_load_lens=[1, 2, 3, 4, 8, 16],
            reuse_read=M.ReuseType(
                req="must",
                levels=[4],
                scope="shared",
            ),
            reuse_write=M.ReuseType(
                req="must",
                levels=[3],
                scope="local",
            ),
        ),
        M.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
        M.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        ),
        M.AutoBind(
            max_threadblocks=256,
            thread_extents=[32, 64, 128, 256, 512, 1024],
        ),
    ]


def default_postprocs():
    from tvm.meta_schedule import postproc as M

    return [
        M.DisallowDynamicLoop(),
        M.RewriteCooperativeFetch(),
        M.RewriteUnboundBlock(),
        M.RewriteParallelVectorizeUnroll(),
        M.RewriteReductionBlock(),
        M.RewriteTensorize(),
        M.VerifyGPUCode(),
    ]


# find out the search space
def find_search_space(m1, n1, p1):
    a, b, c, m, n, p = matmul.params
    context = ms.TuneContext(
        mod=matmul.specialize({m: m1, n: n1, p: p1}),
        target=tvm.target.Target("nvidia/geforce-rtx-3090"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=default_schedule_rules(),
        postprocs=default_postprocs(),
    )
    design_spaces = context.generate_design_space()
    for i, sch in enumerate(design_spaces):
        print(f"design space: {i}")
        print(sch.mod.script())
        print(sch.trace)
        print()


if __name__ == "__main__":
    find_search_space(256, 256, 256)
