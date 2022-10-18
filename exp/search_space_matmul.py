import argparse

import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _
from tvm.tir.tensor_intrin.cuda import get_wmma_store_intrin


# fixed shape matmul
@T.prim_func
def matmulStatic(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (1024, 1024), "float16")
    B = T.match_buffer(b, (1024, 1024), "float16")
    C = T.match_buffer(c, (1024, 1024), "float16")
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# we handle all the shapes of multiple of 16
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    A = T.match_buffer(a, (4096 * m, 4096 * n), "float16")
    B = T.match_buffer(b, (4096 * n, 4096 * p), "float16")
    C = T.match_buffer(c, (4096 * m, 4096 * p), "float16")
    for i, j, k in T.grid(4096 * m, 4096 * n, 4096 * p):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def schedule_matmul(sch: tir.Schedule) -> None:
    b_C = sch.get_block("C")
    sch.transform_layout(
        block=b_C,
        buffer=("read", 1),
        index_map=lambda vj, vk: (
            vk,
            vj,
        ),
    )
    i, j, k = sch.get_loops(block=b_C)
    i0, i1 = sch.split(loop=i, factors=[None, 16])
    j0, j1 = sch.split(loop=j, factors=[None, 16])
    k0, k1 = sch.split(loop=k, factors=[None, 16])
    sch.reorder(i0, j0, k0, i1, j1, k1)
    b_mm = sch.blockize(i1)
    # loop max size is m, n, p
    l_g1, l_g2, l_g3 = sch.get_loops(b_mm)
    cand, prob = categ(8)
    # split l_g1
    v1_g1 = sch.sample_categorical(candidates=cand, probs=prob)
    l_g11, l_g1r = sch.split(loop=l_g1, factors=[None, v1_g1])
    res = sch.sample_perfect_tile(loop=l_g1r, n=3, max_innermost_factor=4)
    l_g12, l_g13, l_g14 = sch.split(loop=l_g1r, factors=[*res])
    # split l_g2
    v1_g2 = sch.sample_categorical(candidates=cand, probs=prob)
    l_g21, l_g2r = sch.split(loop=l_g2, factors=[None, v1_g2])
    res = sch.sample_perfect_tile(loop=l_g2r, n=3, max_innermost_factor=4)
    l_g22, l_g23, l_g24 = sch.split(loop=l_g2r, factors=[*res])
    # split l_g3
    v1_g3 = sch.sample_categorical(candidates=cand, probs=prob)
    l_g31, l_g3r = sch.split(loop=l_g3, factors=[None, v1_g3])
    res = sch.sample_perfect_tile(loop=l_g3r, n=2, max_innermost_factor=4)
    l_g32, l_g33 = sch.split(loop=l_g3r, factors=[*res])
    # only l_g11, l_g21, l_g31 are dynamic loops
    sch.reorder(l_g11, l_g21, l_g12, l_g22, l_g31, l_g32, l_g13, l_g23, l_g33, l_g14, l_g24)
    l_g1 = l_g11
    l_g2 = l_g21
    l_g3 = sch.fuse(l_g12, l_g22)
    sch.bind(loop=l_g1, thread_axis="blockIdx.y")
    sch.bind(loop=l_g2, thread_axis="blockIdx.x")
    sch.bind(loop=l_g3, thread_axis="threadIdx.y")
    # cache write
    C_shared = sch.cache_write(b_mm, 0, "shared")
    sch.reverse_compute_at(C_shared, l_g2)
    C_local = sch.cache_write(b_mm, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_local, l_g3)
    # cooperative fetch for shared memory
    cand, prob = categ(3)  # sample the vectorize size
    _, _, _, l_s = sch.get_loops(block=C_shared)
    v_s = sch.sample_categorical(candidates=cand, probs=prob)
    l_s1, l_s2, l_s3, l_s4 = sch.split(loop=l_s, factors=[None, sch.get(l_g3).extent, 32, v_s])
    sch.vectorize(loop=l_s4)
    sch.bind(loop=l_s3, thread_axis="threadIdx.x")
    sch.bind(loop=l_s2, thread_axis="threadIdx.y")
    # split loops for local memory caching when writing to align with tensor core
    _, _, _, l_l1, l_l2 = sch.get_loops(block=C_local)
    l_l11, l_l12 = sch.split(loop=l_l1, factors=[None, 16])
    l_l21, l_l22 = sch.split(loop=l_l2, factors=[None, 16])
    sch.reorder(l_l21, l_l12)
    # tensorize
    sch.tensorize(block_or_loop=l_l12, tensor_intrin="wmma_store_16x16x16_f16_shared")
    # cache read
    A_shared = sch.cache_read(block=b_mm, read_buffer_index=0, storage_scope="shared")
    B_shared = sch.cache_read(block=b_mm, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=A_shared, loop=l_g31)
    sch.compute_at(block=B_shared, loop=l_g31)
    # fusion for scheduling cooperative fetch
    _, _, _, _, l_s11, l_s12 = sch.get_loops(A_shared)
    l_s1f = sch.fuse(l_s11, l_s12)
    _, _, _, _, l_s21, l_s22 = sch.get_loops(B_shared)
    l_s2f = sch.fuse(l_s21, l_s22)
    # cooperative fetch for shared memory for A
    _, _, _, _, l_s = sch.get_loops(block=A_shared)
    v_s = sch.sample_categorical(candidates=cand, probs=prob)
    l_s1, l_s2, l_s3, l_s4 = sch.split(loop=l_s, factors=[None, sch.get(l_g3).extent, 32, v_s])
    sch.vectorize(loop=l_s4)
    sch.bind(loop=l_s3, thread_axis="threadIdx.x")
    sch.bind(loop=l_s2, thread_axis="threadIdx.y")
    # cooperative fetch for shared memory for B
    _, _, _, _, l_s = sch.get_loops(block=B_shared)
    v_s = sch.sample_categorical(candidates=cand, probs=prob)
    l_s1, l_s2, l_s3, l_s4 = sch.split(loop=l_s, factors=[None, sch.get(l_g3).extent, 32, v_s])
    sch.vectorize(loop=l_s4)
    sch.bind(loop=l_s3, thread_axis="threadIdx.x")
    sch.bind(loop=l_s2, thread_axis="threadIdx.y")
    # cache read for wmma
    A_local = sch.cache_read(block=b_mm, read_buffer_index=0, storage_scope="wmma.matrix_a")
    B_local = sch.cache_read(block=b_mm, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(block=A_local, loop=l_g32)
    sch.compute_at(block=B_local, loop=l_g32)
    # split loops for local memory caching when reading to align with tensor core for A
    _, _, _, _, _, l_l1, l_l2 = sch.get_loops(block=A_local)
    l_l11, l_l12 = sch.split(loop=l_l1, factors=[None, 16])
    l_l21, l_l22 = sch.split(loop=l_l2, factors=[None, 16])
    sch.reorder(l_l21, l_l12)
    # tensorize
    sch.tensorize(block_or_loop=l_l12, tensor_intrin="wmma_load_16x16x16_f16_a")
    # split loops for local memory caching when reading to align with tensor core for B
    _, _, _, _, _, l_l1, l_l2 = sch.get_loops(block=B_local)
    l_l11, l_l12 = sch.split(loop=l_l1, factors=[None, 16])
    l_l21, l_l22 = sch.split(loop=l_l2, factors=[None, 16])
    sch.reorder(l_l21, l_l12)
    # tensorize
    sch.tensorize(block_or_loop=l_l12, tensor_intrin="wmma_load_16x16x16_f16_b")
    # unrolling max step
    # decompose reduction
    b_Co = sch.get_block("C_o")
    sch.decompose_reduction(block=b_Co, loop=l_g31)
    # tensorize
    b_ti = sch.get_block("C_init")
    l_ti, _ = sch.get_loops(block=b_ti)
    sch.tensorize(block_or_loop=l_ti, tensor_intrin="wmma_fill_16x16x16_f16")
    b_tc = sch.get_block("C")
    l_tc, _, _ = sch.get_loops(block=b_tc)
    sch.tensorize(block_or_loop=l_tc, tensor_intrin="wmma_sync_16x16x16_f16f16f16_trans")


def apply_trace(sch):
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    b2 = sch.reindex(block=b0, buffer=("write", 0))
    b3 = sch.reindex(block=b0, buffer=("read", 0))
    b4 = sch.reindex(block=b0, buffer=("read", 1))
    sch.transform_layout(
        block=b0,
        buffer=("read", 0),
        index_map=lambda vi, vk: (
            vi,
            vk,
        ),
    )
    sch.transform_layout(
        block=b0,
        buffer=("read", 1),
        index_map=lambda vj, vk: (
            vk,
            vj,
        ),
    )
    sch.transform_layout(
        block=b0,
        buffer=("write", 0),
        index_map=lambda vi, vj: (
            vi,
            vj,
        ),
    )
    sch.transform_block_layout(
        block=b2,
        index_map=lambda vi, vj, vk: (
            vi,
            vj,
            vk,
        ),
    )
    sch.transform_block_layout(
        block=b3,
        index_map=lambda vi, vj, vk: (
            vi,
            vj,
            vk,
        ),
    )
    sch.transform_block_layout(
        block=b4,
        index_map=lambda vi, vj, vk: (
            vi,
            vj,
            vk,
        ),
    )
    sch.transform_block_layout(
        block=b0,
        index_map=lambda vi, vj, vk: (
            vi,
            vj,
            vk,
        ),
    )
    l5, l6, l7 = sch.get_loops(block=b0)  # from outer to inner
    l8, l9 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)
    l10, l11 = sch.split(loop=l6, factors=[None, 16], preserve_unit_iters=True)
    l12, l13 = sch.split(loop=l5, factors=[None, 16], preserve_unit_iters=True)
    l14, l15, l16, l17, l18, l19 = sch.get_loops(block=b0)
    sch.reorder(l16, l18, l13, l11, l9)
    b20 = sch.blockize(loop=l13)
    sch.annotate(
        block_or_loop=b20,
        ann_key="meta_schedule.auto_tensorize",
        ann_val="wmma_sync_16x16x16_f16f16f16",
    )
    sch.annotate(
        block_or_loop=b20,
        ann_key="meta_schedule.auto_tensorize_init",
        ann_val="wmma_fill_16x16x16_f16",
    )
    sch.annotate(block_or_loop=b20, ann_key="warp_execution", ann_val=1)
    l21, l22, l23 = sch.get_loops(block=b20)
    v24, v25, v26, v27, v28 = sch.sample_perfect_tile(
        loop=l21, n=5, max_innermost_factor=4, decision=[8, 2, 2, 1, 2]
    )
    l29, l30, l31, l32, l33 = sch.split(
        loop=l21, factors=[v24, v25, v26, v27, v28], preserve_unit_iters=True
    )
    v34, v35, v36, v37, v38 = sch.sample_perfect_tile(
        loop=l22, n=5, max_innermost_factor=4, decision=[2, 4, 2, 2, 2]
    )
    l39, l40, l41, l42, l43 = sch.split(
        loop=l22, factors=[v34, v35, v36, v37, v38], preserve_unit_iters=True
    )
    v44, v45, v46 = sch.sample_perfect_tile(
        loop=l23, n=3, max_innermost_factor=4, decision=[16, 4, 1]
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
    sch.reverse_compute_inline(block=b2)
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
    b73 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b73, loop=l47, preserve_unit_loops=True, index=-1)
    l74, l75, l76, l77, l78, l79 = sch.get_loops(block=b73)
    l80 = sch.fuse(l78, l79, preserve_unit_iters=True)
    v81 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(block_or_loop=b73, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
    b82 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="shared")
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
        ann_val="wmma_load_16x16x16_f16_b",
    )
    sch.compute_inline(block=b3)
    sch.compute_inline(block=b4)
    # TODO: figure out buffer_dim_align
    sch.storage_align(block=b73, buffer_index=0, axis=-2, factor=32, offset=8)
    sch.storage_align(block=b82, buffer_index=0, axis=-2, factor=32, offset=8)
    # TODO: unroll
    v135 = sch.sample_categorical(
        candidates=[0, 16, 64, 512, 1024],
        probs=[
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
            0.20000000000000001,
        ],
        decision=1,
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v135)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch")
    l136, l137, l138, l139 = sch.get_loops(block=b53)
    l140, l141, l142, l143 = sch.split(
        loop=l139, factors=[None, 4, 32, 4], preserve_unit_iters=True
    )
    sch.vectorize(loop=l143)
    sch.bind(loop=l142, thread_axis="threadIdx.x")
    sch.bind(loop=l141, thread_axis="threadIdx.y")
    sch.unannotate(block_or_loop=b73, ann_key="meta_schedule.cooperative_fetch")
    l144, l145, l146, l147, l148 = sch.get_loops(block=b73)
    l149, l150, l151, l152 = sch.split(
        loop=l148, factors=[None, 4, 32, 8], preserve_unit_iters=True
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
    sch.annotate(block_or_loop=l170, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l170, ann_key="pragma_unroll_explicit", ann_val=1)
    l178, l179, l180, l181, l182, l183, l184, l185 = sch.get_loops(block=b164)
    sch.annotate(block_or_loop=l178, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l178, ann_key="pragma_unroll_explicit", ann_val=1)
    l186, l187, l188, l189, l190, l191, l192 = sch.get_loops(block=b165)
    sch.annotate(block_or_loop=l186, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l186, ann_key="pragma_unroll_explicit", ann_val=1)
    l193, l194, l195, l196, l197, l198, l199 = sch.get_loops(block=b166)
    sch.annotate(block_or_loop=l193, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l193, ann_key="pragma_unroll_explicit", ann_val=1)
    l200, l201, l202, l203, l204, l205, l206, l207, l208, l209 = sch.get_loops(block=b167)
    sch.annotate(block_or_loop=l200, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l200, ann_key="pragma_unroll_explicit", ann_val=1)
    l210, l211, l212, l213, l214 = sch.get_loops(block=b168)
    sch.annotate(block_or_loop=l210, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l210, ann_key="pragma_unroll_explicit", ann_val=1)
    l215, l216, l217, l218, l219, l220, l221 = sch.get_loops(block=b169)
    sch.annotate(block_or_loop=l215, ann_key="pragma_auto_unroll_max_step", ann_val=16)
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
    b235 = sch.get_block(name="A_reindex_shared_wmma.matrix_a_o", func_name="main")
    sch.unannotate(block_or_loop=b235, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b235, tensor_intrin="wmma_load_16x16x16_f16_a")
    b236 = sch.get_block(name="B_reindex_shared_wmma.matrix_b_o", func_name="main")
    sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b236, tensor_intrin="wmma_load_16x16x16_f16_b")
    b237 = sch.get_block(name="C_o_update", func_name="main")
    sch.unannotate(block_or_loop=b237, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b237, tensor_intrin="wmma_sync_16x16x16_f16f16f16")
    b238 = sch.get_block(name="C_reindex_shared_wmma.accumulator_o", func_name="main")
    sch.unannotate(block_or_loop=b238, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b238, tensor_intrin="wmma_store_16x16x16_f16_shared")


def categ(k):
    """should tweak weights in the future to account for independence"""
    arr = np.arange(k + 1)
    cand = 2**arr
    cand = [int(i) for i in cand]
    prob = np.ones(k + 1) / (k + 1)
    return cand, list(prob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mine", action="store_true")
    args = parser.parse_args()

    if args.mine:
        sch = tvm.tir.Schedule(matmul)
        schedule_matmul(sch)
        sch.mod.show()
        matmul_mod = tvm.build(sch.mod, target="cuda")
    else:
        sch = tvm.tir.Schedule(matmulStatic)
        apply_trace(sch)
        sch.mod.show()
        matmul_mod = tvm.build(sch.mod, target="cuda")

    """
    print("tensor intrinsic:")
    res = get_wmma_store_intrin(16, 16, 16, "float16", "load")
    res[0].show()

    # evaluate the running time
    dev = tvm.cuda(0)
    A_np = np.random.uniform(size=(1024, )).astype("float32")
    B_np = np.random.uniform(size=(1024, )).astype("float32")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((1024, ), dtype="float32"), dev)
    num_flop = 2 * 1024
    evaluator = vectoradd_mod.time_evaluator("vectoradd", dev, number=10)
    print("vectoradd running time: %f GFLOPS\n" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
    """

    """
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.tune_tir(
            mod=matmulStatic,
            target=target,
            config=ms.TuneConfig(
                num_trials_per_iter=32,
                max_trials_per_task=1000,
                max_trials_global=32,
            ),
            sch_rules=ms.default_config._DefaultCUDATensorCore.schedule_rules,
            postprocs=ms.default_config._DefaultCUDATensorCore.postprocs,
            work_dir="logs/test-1",
        )
        print(sch.trace.show())
    """
