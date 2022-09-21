import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _


@T.prim_func
def vectoradd_const(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "vectoradd_const", "tir.noalias": True})
    A = T.match_buffer(a, (256, ), "float32")
    B = T.match_buffer(b, (256, ), "float32")
    C = T.match_buffer(c, (256, ), "float32")
    for i in T.grid(256):
        with T.block("update"):
            vi = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]


@T.prim_func
def vectoradd(a: T.handle, b: T.handle, c: T.handle) -> None:
    m = T.var("int32")
    A = T.match_buffer(a, (m, ), "float32")
    B = T.match_buffer(b, (m, ), "float32")
    C = T.match_buffer(c, (m, ), "float32")
    for i in T.grid(m):
        with T.block("update"):
            vi = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]


# m, n, p should be multiples of 16
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    m = T.var("int32")
    n = T.var("int32")
    p = T.var("int32")
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

def find_search_space_vectoradd_const():
    context = ms.TuneContext(
        mod=vectoradd_const,
        target=tvm.target.Target("nvidia/geforce-rtx-3090"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=ms.default_config._DefaultCUDA.schedule_rules,
        postprocs=ms.default_config._DefaultCUDA.postprocs,
    )
    design_spaces = context.generate_design_space()
    for i, sch in enumerate(design_spaces):
        print(f"design space: {i}")
        print(sch.mod.script())
        print(sch.trace)
        print()

# find out the search space
def find_search_space_vectoradd():
    context = ms.TuneContext(
        mod=vectoradd,
        target=tvm.target.Target("nvidia/geforce-rtx-3090"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=ms.default_config._DefaultCUDA.schedule_rules,
        postprocs=ms.default_config._DefaultCUDA.postprocs,
    )
    design_spaces = context.generate_design_space()
    for i, sch in enumerate(design_spaces):
        print(f"design space: {i}")
        print(sch.mod.script())
        print(sch.trace)
        print()

# find out the search space
def find_search_space_matmul():
    context = ms.TuneContext(
        mod=matmul,
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

def tune_vectoradd_const():
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.tune_tir(
            mod=vectoradd_const,
            target=target,
            config=ms.TuneConfig(
                num_trials_per_iter=32,
                max_trials_per_task=1000,
                max_trials_global=1000,
            ),
            sch_rules=ms.default_config._DefaultCUDATensorCore.schedule_rules,
            postprocs=ms.default_config._DefaultCUDATensorCore.postprocs,
            work_dir="logs/vectoradd_const",
        )

def search_space_vectoradd_const(sch: tir.Schedule) -> None:
    block_u = sch.get_block("update")
    i, = sch.get_loops(block=block_u)
    i0, i1 = sch.split(i, [None, 64])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")

def search_space_vectoradd(sch: tir.Schedule) -> None:
    pass

def search_space_matmul(sch: tir.Schedule) -> None:
    pass


"""
scheduling template for all matmul (generated by TVM)

# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="update", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  b2 = sch.reindex(block=b0, buffer=("write", 0))
  b3 = sch.reindex(block=b0, buffer=("read", 0))
  b4 = sch.reindex(block=b0, buffer=("read", 1))
  sch.transform_layout(block=b0, buffer=("read", 0), index_map=lambda vi, vk: (vi, vk,))
  sch.transform_layout(block=b0, buffer=("read", 1), index_map=lambda vj, vk: (vk, vj,))  // this can be transposed
  sch.transform_layout(block=b0, buffer=("write", 0), index_map=lambda vi, vj: (vi, vj,))
  sch.transform_block_layout(block=b2, index_map=lambda vi, vj, vk: (vi, vj, vk,))
  sch.transform_block_layout(block=b3, index_map=lambda vi, vj, vk: (vi, vj, vk,))
  sch.transform_block_layout(block=b4, index_map=lambda vi, vj, vk: (vi, vj, vk,))
  sch.transform_block_layout(block=b0, index_map=lambda vi, vj, vk: (vi, vj, vk,))
  l5, l6, l7 = sch.get_loops(block=b0)
  l8, l9 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)
  l10, l11 = sch.split(loop=l6, factors=[None, 16], preserve_unit_iters=True)
  l12, l13 = sch.split(loop=l5, factors=[None, 16], preserve_unit_iters=True)
  l14, l15, l16, l17, l18, l19 = sch.get_loops(block=b0)
  sch.reorder(l16, l18, l13, l11, l9)
  b20 = sch.blockize(loop=l13)
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")  // can use transposed tensor core
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
  sch.annotate(block_or_loop=b20, ann_key="warp_execution", ann_val=1)
  l21, l22, l23 = sch.get_loops(block=b20)
  v24, v25, v26, v27, v28 = sch.sample_perfect_tile(loop=l21, n=5, max_innermost_factor=4, decision=[1, 8, 2, 16, 1])  // decision can vary
  l29, l30, l31, l32, l33 = sch.split(loop=l21, factors=[v24, v25, v26, v27, v28], preserve_unit_iters=True)
  v34, v35, v36, v37, v38 = sch.sample_perfect_tile(loop=l22, n=5, max_innermost_factor=4, decision=[8, 8, 2, 1, 2])  // decision can vary
  l39, l40, l41, l42, l43 = sch.split(loop=l22, factors=[v34, v35, v36, v37, v38], preserve_unit_iters=True)
  v44, v45, v46 = sch.sample_perfect_tile(loop=l23, n=3, max_innermost_factor=4, decision=[2, 32, 4])  // decision can vary
  l47, l48, l49 = sch.split(loop=l23, factors=[v44, v45, v46], preserve_unit_iters=True)
  sch.reorder(l29, l39, l30, l40, l31, l41, l47, l48, l32, l42, l49, l33, l43)
  l50 = sch.fuse(l29, l39, preserve_unit_iters=True)
  sch.bind(loop=l50, thread_axis="blockIdx.y")
  l51 = sch.fuse(l30, l40, preserve_unit_iters=True)
  sch.bind(loop=l51, thread_axis="blockIdx.x")
  l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
  sch.bind(loop=l52, thread_axis="threadIdx.y")
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
  b53 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="shared")
  sch.reverse_compute_at(block=b53, loop=l51, preserve_unit_loops=True, index=-1)
  b54 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="wmma.accumulator")
  sch.reverse_compute_at(block=b54, loop=l52, preserve_unit_loops=True, index=-1)
  v55 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)  // decision can vary
  sch.annotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
  sch.reverse_compute_inline(block=b2)
  l56, l57, l58, l59, l60 = sch.get_loops(block=b54)
  l61, l62 = sch.split(loop=l60, factors=[None, 16], preserve_unit_iters=True)
  l63, l64 = sch.split(loop=l59, factors=[None, 16], preserve_unit_iters=True)
  l65, l66, l67, l68, l69, l70, l71 = sch.get_loops(block=b54)
  sch.reorder(l70, l64, l62)
  b72 = sch.blockize(loop=l64)
  sch.annotate(block_or_loop=b72, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared")
  b73 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="shared")
  sch.compute_at(block=b73, loop=l47, preserve_unit_loops=True, index=-1)
  l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b73)
  l80 = sch.fuse(l78, l79, preserve_unit_iters=True)
  v81 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)  // decision can vary
  sch.annotate(block_or_loop=b73, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
  b82 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="shared")
  sch.compute_at(block=b82, loop=l47, preserve_unit_loops=True, index=-1)
  l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b82)
  l89 = sch.fuse(l87, l88, preserve_unit_iters=True)
  v90 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)  // decision can vary
  sch.annotate(block_or_loop=b82, ann_key="meta_schedule.cooperative_fetch", ann_val=v90)
  b91 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="wmma.matrix_a")
  sch.compute_at(block=b91, loop=l48, preserve_unit_loops=True, index=-1)
  l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b91)
  l99, l100 = sch.split(loop=l98, factors=[None, 16], preserve_unit_iters=True)
  l101, l102 = sch.split(loop=l97, factors=[None, 16], preserve_unit_iters=True)
  l103, l104, l105, l106, l107, l108, l109, l110, l111 = sch.get_loops(block=b91)
  sch.reorder(l110, l102, l100)
  b112 = sch.blockize(loop=l102)
  sch.annotate(block_or_loop=b112, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a")
  b113 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="wmma.matrix_b")
  sch.compute_at(block=b113, loop=l48, preserve_unit_loops=True, index=-1)
  l114, l115, l116, l117, l118, l119, l120 = sch.get_loops(block=b113)
  l121, l122 = sch.split(loop=l120, factors=[None, 16], preserve_unit_iters=True)
  l123, l124 = sch.split(loop=l119, factors=[None, 16], preserve_unit_iters=True)
  l125, l126, l127, l128, l129, l130, l131, l132, l133 = sch.get_loops(block=b113)
  sch.reorder(l132, l124, l122)
  b134 = sch.blockize(loop=l124)
  sch.annotate(block_or_loop=b134, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b")  // can use transposed tensor core
  sch.compute_inline(block=b3)
  sch.compute_inline(block=b4)
  sch.storage_align(block=b73, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.storage_align(block=b82, buffer_index=0, axis=-2, factor=32, offset=8)
  v135 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)  // decision can vary
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v135)
"""
