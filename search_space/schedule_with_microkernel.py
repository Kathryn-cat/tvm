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


@T.prim_func
def microkernelOrig(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "microkernelOrig", "tir.noalias": True})
    A = T.match_buffer(a, (2048, 512), "float16")
    B = T.match_buffer(b, (512, 2048), "float16")
    C = T.match_buffer(c, (2048, 2048), "float16")
    for i, j, k in T.grid(2048, 2048, 512):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def matmul1(a: T.handle, b: T.handle, c: T.handle, k: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul1", "tir.noalias": True})
    A = T.match_buffer(a, (256 * k, 768), "float16")
    B = T.match_buffer(b, (768, 2304), "float16")
    C = T.match_buffer(c, (256 * k, 2304), "float16")
    for i, j, k in T.grid(256 * k, 2304, 768):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# microkernel is 64, 128, 32
def apply_trace(sch: tir.Schedule) -> None:
    C = sch.get_block("C")
    i, j, k = sch.get_loops(C)
    i0, i1 = sch.split(i, [None, 64])
    j0, j1 = sch.split(j, [None, 128])
    sch.reorder(i0, j0, i1, j1, k)
    sch.blockize(i1)
    k0, k1 = sch.split(k, [None, 32])
    sch.reorder(k0, i1, j1, k1)
    CTA = sch.blockize(i1)

    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    # b2 = sch.reindex(block=b0, buffer=("write", 0))
    # b3 = sch.reindex(block=b0, buffer=("read", 0))
    # b4 = sch.reindex(block=b0, buffer=("read", 1))
    # sch.transform_layout(
    #    block=b0,
    #    buffer=("read", 0),
    #    index_map=lambda vi, vk: (
    #        vi,
    #        vk,
    #    ),
    #    pad_value=None,
    # )
    # sch.transform_layout(
    #    block=b0,
    #    buffer=("read", 1),
    #    index_map=lambda vj, vk: (
    #        vj,
    #        vk,
    #    ),
    #    pad_value=None,
    # )
    # sch.transform_layout(
    #    block=b0,
    #    buffer=("write", 0),
    #    index_map=lambda vi, vj: (
    #        vi,
    #        vj,
    #    ),
    #    pad_value=None,
    # )
    # sch.transform_block_layout(
    #    block=b2,
    #    index_map=lambda vi, vj: (
    #        vi,
    #        vj,
    #    ),
    # )
    # sch.transform_block_layout(
    #    block=b3,
    #    index_map=lambda vi, vk: (
    #        vi,
    #        vk,
    #    ),
    # )
    # sch.transform_block_layout(
    #    block=b4,
    #    index_map=lambda vj, vk: (
    #        vj,
    #        vk,
    #    ),
    # )
    # sch.transform_block_layout(
    #    block=b0,
    #    index_map=lambda vi, vj, vk: (
    #        vi,
    #        vj,
    #        vk,
    #    ),
    # )
    l5, l6, l7 = sch.get_loops(block=b0)
    l8, l9 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)
    l10, l11 = sch.split(loop=l6, factors=[None, 16], preserve_unit_iters=True)
    l12, l13 = sch.split(loop=l5, factors=[None, 16], preserve_unit_iters=True)
    l14, l15, l16, l17, l18, l19 = sch.get_loops(block=b0)
    sch.reorder(l16, l18, l13, l11, l9)
    l14_1, l15_1, l16_1, l17_1, l18_1, l19_1 = sch.get_loops(block=b0)  # refer to these loops
    b20 = sch.blockize(loop=l17_1)
    """
    sch.annotate(
        block_or_loop=b20,
        ann_key="meta_schedule.auto_tensorize",
        ann_val="wmma_sync_16x16x16_f16f16f16_trans",
    )
    sch.annotate(
        block_or_loop=b20,
        ann_key="meta_schedule.auto_tensorize_init",
        ann_val="wmma_fill_16x16x16_f16",
    )
    sch.annotate(block_or_loop=b20, ann_key="warp_execution", ann_val=1)
    l21, l22, l23 = sch.get_loops(block=b20)
    v24, v25, v26, v27, v28 = sch.sample_perfect_tile(
        loop=l21, n=5, max_innermost_factor=4, decision=[2, 16, 2, 2, 1]  # TODO: 128
    )
    l29, l30, l31, l32, l33 = sch.split(
        loop=l21, factors=[v24, v25, v26, v27, v28], preserve_unit_iters=True
    )
    v34, v35, v36, v37, v38 = sch.sample_perfect_tile(
        loop=l22, n=5, max_innermost_factor=4, decision=[16, 1, 2, 4, 1]  # TODO: 128
    )
    l39, l40, l41, l42, l43 = sch.split(
        loop=l22, factors=[v34, v35, v36, v37, v38], preserve_unit_iters=True
    )
    v44, v45, v46 = sch.sample_perfect_tile(
        loop=l23, n=3, max_innermost_factor=4, decision=[16, 1, 2]  # TODO: 32
    )
    l47, l48, l49 = sch.split(loop=l23, factors=[v44, v45, v46], preserve_unit_iters=True)
    sch.reorder(l29, l39, l30, l40, l31, l41, l47, l48, l32, l42, l49, l33, l43)
    l50 = sch.fuse(l29, l39, preserve_unit_iters=True)
    sch.bind(loop=l50, thread_axis="blockIdx.y")
    l51 = sch.fuse(l30, l40, preserve_unit_iters=True)
    sch.bind(loop=l51, thread_axis="blockIdx.x")
    l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
    sch.bind(loop=l52, thread_axis="threadIdx.y")
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(
        block_or_loop=b20, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
    )
    b53 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="shared")
    sch.reverse_compute_at(block=b53, loop=l51, preserve_unit_loops=True, index=-1)
    b54 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(block=b54, loop=l52, preserve_unit_loops=True, index=-1)
    v55 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2
    )
    sch.annotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
    # sch.reverse_compute_inline(block=b2)
    l56, l57, l58, l59, l60 = sch.get_loops(block=b54)
    l61, l62 = sch.split(loop=l60, factors=[None, 16], preserve_unit_iters=True)
    l63, l64 = sch.split(loop=l59, factors=[None, 16], preserve_unit_iters=True)
    l65, l66, l67, l68, l69, l70, l71 = sch.get_loops(block=b54)
    sch.reorder(l70, l64, l62)
    b72 = sch.blockize(loop=l64)
    sch.annotate(
        block_or_loop=b72,
        ann_key="meta_schedule.auto_tensorize",
        ann_val="wmma_store_16x16x16_f16_shared",
    )
    b73 = sch.cache_read(
        block=b20, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b20]
    )
    sch.compute_at(block=b73, loop=l47, preserve_unit_loops=True, index=-1)
    l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b73)
    l80 = sch.fuse(l78, l79, preserve_unit_iters=True)
    v81 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1
    )
    sch.annotate(block_or_loop=b73, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
    b82 = sch.cache_read(
        block=b20, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b20]
    )
    sch.compute_at(block=b82, loop=l47, preserve_unit_loops=True, index=-1)
    l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b82)
    l89 = sch.fuse(l87, l88, preserve_unit_iters=True)
    v90 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2
    )
    sch.annotate(block_or_loop=b82, ann_key="meta_schedule.cooperative_fetch", ann_val=v90)
    b91 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="wmma.matrix_a")
    sch.compute_at(block=b91, loop=l48, preserve_unit_loops=True, index=-1)
    l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b91)
    l99, l100 = sch.split(loop=l98, factors=[None, 16], preserve_unit_iters=True)
    l101, l102 = sch.split(loop=l97, factors=[None, 16], preserve_unit_iters=True)
    l103, l104, l105, l106, l107, l108, l109, l110, l111 = sch.get_loops(block=b91)
    sch.reorder(l110, l102, l100)
    b112 = sch.blockize(loop=l102)
    sch.annotate(
        block_or_loop=b112,
        ann_key="meta_schedule.auto_tensorize",
        ann_val="wmma_load_16x16x16_f16_a",
    )
    b113 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(block=b113, loop=l48, preserve_unit_loops=True, index=-1)
    l114, l115, l116, l117, l118, l119, l120 = sch.get_loops(block=b113)
    l121, l122 = sch.split(loop=l120, factors=[None, 16], preserve_unit_iters=True)
    l123, l124 = sch.split(loop=l119, factors=[None, 16], preserve_unit_iters=True)
    l125, l126, l127, l128, l129, l130, l131, l132, l133 = sch.get_loops(block=b113)
    sch.reorder(l132, l124, l122)
    b134 = sch.blockize(loop=l124)
    sch.annotate(
        block_or_loop=b134,
        ann_key="meta_schedule.auto_tensorize",
        ann_val="wmma_load_16x16x16_f16_b_trans",
    )
    # sch.compute_inline(block=b3)
    # sch.compute_inline(block=b4)
    sch.storage_align(block=b73, buffer_index=0, axis=-2, factor=32, offset=8)
    sch.storage_align(block=b82, buffer_index=0, axis=-2, factor=32, offset=8)
    v135 = sch.sample_categorical(
        candidates=[0, 16, 64, 512, 1024],
        probs=[
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
        ],
        decision=3,
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v135)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch")
    l136, l137, l138, l139 = sch.get_loops(block=b53)
    l140, l141, l142, l143 = sch.split(
        loop=l139, factors=[None, 4, 32, 1], preserve_unit_iters=True
    )
    sch.vectorize(loop=l143)
    sch.bind(loop=l142, thread_axis="threadIdx.x")
    sch.bind(loop=l141, thread_axis="threadIdx.y")
    sch.unannotate(block_or_loop=b73, ann_key="meta_schedule.cooperative_fetch")
    l144, l145, l146, l147, l148 = sch.get_loops(block=b73)
    l149, l150, l151, l152 = sch.split(
        loop=l148, factors=[None, 4, 32, 2], preserve_unit_iters=True
    )
    sch.vectorize(loop=l152)
    sch.bind(loop=l151, thread_axis="threadIdx.x")
    sch.bind(loop=l150, thread_axis="threadIdx.y")
    sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.cooperative_fetch")
    l153, l154, l155, l156, l157 = sch.get_loops(block=b82)
    l158, l159, l160, l161 = sch.split(
        loop=l157, factors=[None, 4, 32, 4], preserve_unit_iters=True
    )
    sch.vectorize(loop=l161)
    sch.bind(loop=l160, thread_axis="threadIdx.x")
    sch.bind(loop=l159, thread_axis="threadIdx.y")
    b162 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b162, ann_key="meta_schedule.unroll_explicit")
    b163, b164, b165, b166, b167, b168, b169 = sch.get_child_blocks(b162)
    l170, l171, l172, l173, l174, l175, l176, l177 = sch.get_loops(block=b163)
    sch.annotate(block_or_loop=l170, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l170, ann_key="pragma_unroll_explicit", ann_val=1)
    l178, l179, l180, l181, l182, l183, l184, l185 = sch.get_loops(block=b164)
    sch.annotate(block_or_loop=l178, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l178, ann_key="pragma_unroll_explicit", ann_val=1)
    l186, l187, l188, l189, l190, l191, l192 = sch.get_loops(block=b165)
    sch.annotate(block_or_loop=l186, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l186, ann_key="pragma_unroll_explicit", ann_val=1)
    l193, l194, l195, l196, l197, l198, l199 = sch.get_loops(block=b166)
    sch.annotate(block_or_loop=l193, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l193, ann_key="pragma_unroll_explicit", ann_val=1)
    l200, l201, l202, l203, l204, l205, l206, l207, l208, l209 = sch.get_loops(block=b167)
    sch.annotate(block_or_loop=l200, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l200, ann_key="pragma_unroll_explicit", ann_val=1)
    l210, l211, l212, l213, l214 = sch.get_loops(block=b168)
    sch.annotate(block_or_loop=l210, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l210, ann_key="pragma_unroll_explicit", ann_val=1)
    l215, l216, l217, l218, l219, l220, l221 = sch.get_loops(block=b169)
    sch.annotate(block_or_loop=l215, ann_key="pragma_auto_unroll_max_step", ann_val=512)
    sch.annotate(block_or_loop=l215, ann_key="pragma_unroll_explicit", ann_val=1)
    b222 = sch.get_block(name="C_o", func_name="main")
    l223, l224, l225, l226, l227, l228, l229, l230, l231, l232 = sch.get_loops(block=b222)
    b233 = sch.decompose_reduction(block=b222, loop=l226)
    sch.unannotate(block_or_loop=b233, ann_key="meta_schedule.auto_tensorize")
    sch.annotate(
        block_or_loop=b233, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16"
    )
    sch.unannotate(block_or_loop=b222, ann_key="meta_schedule.auto_tensorize_init")
    sch.unannotate(block_or_loop=b233, ann_key="meta_schedule.auto_tensorize_init")
    b234 = sch.get_block(name="C_o_init", func_name="main")
    sch.unannotate(block_or_loop=b234, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b234, tensor_intrin="wmma_fill_16x16x16_f16")
    # b235 = sch.get_block(name="A_reindex_shared_wmma.matrix_a_o", func_name="main")
    b235 = sch.get_block(name="A_shared_wmma.matrix_a_o", func_name="main")
    sch.unannotate(block_or_loop=b235, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b235, tensor_intrin="wmma_load_16x16x16_f16_a")
    # b236 = sch.get_block(name="B_reindex_shared_wmma.matrix_b_o", func_name="main")
    b236 = sch.get_block(name="B_shared_wmma.matrix_b_o", func_name="main")
    sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b236, tensor_intrin="wmma_load_16x16x16_f16_b_trans")
    b237 = sch.get_block(name="C_o_update", func_name="main")
    sch.unannotate(block_or_loop=b237, ann_key="meta_schedule.auto_tensorize")
    # sch.tensorize(block_or_loop=b237, tensor_intrin="wmma_sync_16x16x16_f16f16f16_trans")
    sch.tensorize(block_or_loop=b237, tensor_intrin="wmma_sync_16x16x16_f16f16f16")
    # b238 = sch.get_block(name="C_reindex_shared_wmma.accumulator_o", func_name="main")
    b238 = sch.get_block(name="C_shared_wmma.accumulator_o", func_name="main")
    sch.unannotate(block_or_loop=b238, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b238, tensor_intrin="wmma_store_16x16x16_f16_shared")
    """


if __name__ == "__main__":
    sch = tir.Schedule(microkernelOrig)
    apply_trace(sch)
    sch.mod.show()
