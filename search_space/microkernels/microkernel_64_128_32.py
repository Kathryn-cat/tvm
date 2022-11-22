from tvm.script import tir as T


@tvm.script.ir_module
class Microkernel_64_128_32:
    @T.prim_func
    def main(
        A: T.Buffer[(64, 32), "float16"],
        B: T.Buffer[(32, 128), "float16"],
        C: T.Buffer[(64, 128), "float16"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "microkernel_64_128_32", "tir.noalias": True})
        a0 = T.var("int32")
        a1 = T.var("int32")
        b0 = T.var("int32")
        b1 = T.var("int32")
        c0 = T.var("int32")
        c1 = T.var("int32")
        d0 = T.var("int32")
        d0_1 = T.var("int32")
        d0_2 = T.var("int32")
        d0_3 = T.var("int32")
        d1 = T.var("int32")
        d1_1 = T.var("int32")
        d1_2 = T.var("int32")
        d1_3 = T.var("int32")
        s0 = T.var("int32")
        s0_1 = T.var("int32")
        s0_2 = T.var("int32")
        s1 = T.var("int32")
        s1_1 = T.var("int32")
        s1_2 = T.var("int32")
        # body
        # with T.block("root")
        C_reindex_shared = T.alloc_buffer([64, 128], dtype="float16", scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer(
            [64, 128], dtype="float16", scope="wmma.accumulator"
        )
        A_reindex_shared = T.alloc_buffer([64, 32], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([32, 128], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer(
            [64, 32], dtype="float16", scope="wmma.matrix_a"
        )
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer(
            [32, 128], dtype="float16", scope="wmma.matrix_b"
        )
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(
            4,
            thread="blockIdx.y",
            annotations={"pragma_auto_unroll_max_step": 1024, "pragma_unroll_explicit": 1},
        ):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(8, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(1, thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(
                        1, 1, 1, 1
                    ):
                        with T.block("C_o_init"):
                            v0_o = T.axis.spatial(
                                4, ax0_0_3_init + ax0_0_4_init + ax0_0_0_ax1_0_0_fused
                            )
                            v1_o = T.axis.spatial(
                                8, ax1_0_3_init + ax1_0_4_init + ax0_0_1_ax1_0_1_fused
                            )
                            T.reads()
                            T.writes(
                                C_reindex_shared_wmma_accumulator[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ]
                            )
                            T.block_attr(
                                {
                                    "meta_schedule.thread_extent_high_inclusive": 1024,
                                    "meta_schedule.thread_extent_low_inclusive": 1,
                                    "warp_execution": 1,
                                }
                            )
                            C_1 = T.match_buffer(
                                C_reindex_shared_wmma_accumulator[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ],
                                [16, 16],
                                dtype="float16",
                                strides=[d1, d0],
                                scope="wmma.accumulator",
                                offset_factor=16,
                            )
                            T.evaluate(
                                T.tvm_fill_fragment(
                                    C_1.data,
                                    16,
                                    16,
                                    16,
                                    C_1.elem_offset // d1 // 16 * (d1 // 16)
                                    + C_1.elem_offset % d1 // 16,
                                    T.float32(0),
                                    dtype="handle",
                                )
                            )
                    for ax2_0_0 in T.serial(1):
                        for ax0_ax1_fused_0 in T.serial(8):
                            for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(2):
                                        with T.block("A_reindex_shared"):
                                            v0 = T.axis.spatial(
                                                64,
                                                ax0_0_0_ax1_0_0_fused * 16
                                                + (
                                                    ax0_ax1_fused_0 * 64
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                // 32,
                                            )
                                            v1 = T.axis.spatial(
                                                32,
                                                (
                                                    ax0_ax1_fused_0 * 64
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                % 32,
                                            )
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(2):
                            for ax0_ax1_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(8):
                                        with T.block("B_reindex_shared"):
                                            v0 = T.axis.spatial(
                                                32,
                                                (
                                                    ax0_ax1_fused_0 * 256
                                                    + ax0_ax1_fused_1 * 256
                                                    + ax0_ax1_fused_2 * 8
                                                    + ax0_ax1_fused_3
                                                )
                                                // 16,
                                            )
                                            v1 = T.axis.spatial(
                                                128,
                                                ax0_0_1_ax1_0_1_fused * 16
                                                + (
                                                    ax0_ax1_fused_0 * 256
                                                    + ax0_ax1_fused_1 * 256
                                                    + ax0_ax1_fused_2 * 8
                                                    + ax0_ax1_fused_3
                                                )
                                                % 16,
                                            )
                                            T.reads(B[v0, v1])
                                            T.writes(B_reindex_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(1):
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax0_0_0_ax1_0_0_fused, ax1_0])
                                    T.reads(
                                        A_reindex_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.writes(
                                        A_reindex_shared_wmma_matrix_a[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    A_1 = T.match_buffer(
                                        A_reindex_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[s1, s0],
                                        scope="shared",
                                        offset_factor=16,
                                    )
                                    C_2 = T.match_buffer(
                                        A_reindex_shared_wmma_matrix_a[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[d1_1, d0_1],
                                        scope="wmma.matrix_a",
                                        offset_factor=16,
                                    )
                                    T.evaluate(
                                        T.tvm_load_matrix_sync(
                                            C_2.data,
                                            16,
                                            16,
                                            16,
                                            C_2.elem_offset // d1_1 // 16 * (d1_1 // 16)
                                            + C_2.elem_offset % d1_1 // 16,
                                            T.tvm_access_ptr(
                                                T.type_annotation(dtype="float16"),
                                                A_1.data,
                                                A_1.elem_offset,
                                                s1 * 16,
                                                1,
                                                dtype="handle",
                                            ),
                                            s1,
                                            "row_major",
                                            dtype="handle",
                                        )
                                    )
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax0_0, ax0_0_1_ax1_0_1_fused])
                                    T.reads(
                                        B_reindex_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.writes(
                                        B_reindex_shared_wmma_matrix_b[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    A_2 = T.match_buffer(
                                        B_reindex_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[s1_1, s0_1],
                                        scope="shared",
                                        offset_factor=16,
                                    )
                                    C_3 = T.match_buffer(
                                        B_reindex_shared_wmma_matrix_b[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[d1_2, d0_2],
                                        scope="wmma.matrix_b",
                                        offset_factor=16,
                                    )
                                    T.evaluate(
                                        T.tvm_load_matrix_sync(
                                            C_3.data,
                                            16,
                                            16,
                                            16,
                                            C_3.elem_offset // d1_2 // 16 * (d1_2 // 16)
                                            + C_3.elem_offset % d1_2 // 16,
                                            T.tvm_access_ptr(
                                                T.type_annotation(dtype="float16"),
                                                A_2.data,
                                                A_2.elem_offset,
                                                s1_1 * 16,
                                                1,
                                                dtype="handle",
                                            ),
                                            s1_1,
                                            "row_major",
                                            dtype="handle",
                                        )
                                    )
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(
                                1, 1, 2, 1, 1
                            ):
                                with T.block("C_o_update"):
                                    v0_o = T.axis.spatial(
                                        4, ax0_0_3 + ax0_0_4 + ax0_0_0_ax1_0_0_fused
                                    )
                                    v1_o = T.axis.spatial(
                                        8, ax1_0_3 + ax1_0_4 + ax0_0_1_ax1_0_1_fused
                                    )
                                    v2_o = T.axis.reduce(2, ax2_0_0 * 2 + ax2_0_1 * 2 + ax2_0_2)
                                    T.reads(
                                        C_reindex_shared_wmma_accumulator[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        A_reindex_shared_wmma_matrix_a[
                                            v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                                        ],
                                        B_reindex_shared_wmma_matrix_b[
                                            v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                    )
                                    T.writes(
                                        C_reindex_shared_wmma_accumulator[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.block_attr(
                                        {
                                            "meta_schedule.thread_extent_high_inclusive": 1024,
                                            "meta_schedule.thread_extent_low_inclusive": 1,
                                            "warp_execution": 1,
                                        }
                                    )
                                    A_3 = T.match_buffer(
                                        A_reindex_shared_wmma_matrix_a[
                                            v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[a1, a0],
                                        scope="wmma.matrix_a",
                                        offset_factor=16,
                                    )
                                    B_1 = T.match_buffer(
                                        B_reindex_shared_wmma_matrix_b[
                                            v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[b1, b0],
                                        scope="wmma.matrix_b",
                                        offset_factor=16,
                                    )
                                    C_4 = T.match_buffer(
                                        C_reindex_shared_wmma_accumulator[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[c1, c0],
                                        scope="wmma.accumulator",
                                        offset_factor=16,
                                    )
                                    T.evaluate(
                                        T.tvm_mma_sync(
                                            C_4.data,
                                            C_4.elem_offset // c1 // 16 * (c1 // 16)
                                            + C_4.elem_offset % c1 // 16,
                                            A_3.data,
                                            A_3.elem_offset // a1 // 16 * (a1 // 16)
                                            + A_3.elem_offset % a1 // 16,
                                            B_1.data,
                                            B_1.elem_offset // b1 // 16 * (b1 // 16)
                                            + B_1.elem_offset % b1 // 16,
                                            C_4.data,
                                            C_4.elem_offset // c1 // 16 * (c1 // 16)
                                            + C_4.elem_offset % c1 // 16,
                                            dtype="handle",
                                        )
                                    )
                    for ax0_0, ax1_0 in T.grid(1, 1):
                        with T.block("C_reindex_shared_wmma.accumulator_o"):
                            v0_o, v1_o = T.axis.remap(
                                "SS", [ax0_0_0_ax1_0_0_fused, ax0_0_1_ax1_0_1_fused]
                            )
                            T.reads(
                                C_reindex_shared_wmma_accumulator[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ]
                            )
                            T.writes(
                                C_reindex_shared[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ]
                            )
                            A_4 = T.match_buffer(
                                C_reindex_shared_wmma_accumulator[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ],
                                [16, 16],
                                dtype="float16",
                                strides=[d1_3, d0_3],
                                scope="wmma.accumulator",
                                offset_factor=16,
                            )
                            C_5 = T.match_buffer(
                                C_reindex_shared[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ],
                                [16, 16],
                                dtype="float16",
                                strides=[s1_2, s0_2],
                                scope="shared",
                                offset_factor=16,
                            )
                            T.evaluate(
                                T.tvm_store_matrix_sync(
                                    A_4.data,
                                    16,
                                    16,
                                    16,
                                    A_4.elem_offset // d1_3 // 16 * (d1_3 // 16)
                                    + A_4.elem_offset % d1_3 // 16,
                                    T.tvm_access_ptr(
                                        T.type_annotation(dtype="float16"),
                                        C_5.data,
                                        C_5.elem_offset,
                                        s1_2 * 16,
                                        2,
                                        dtype="handle",
                                    ),
                                    s1_2,
                                    "row_major",
                                    dtype="handle",
                                )
                            )
                for ax0, ax1_0 in T.grid(16, 1):
                    for ax1_1 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax1_2 in T.thread_binding(32, thread="threadIdx.x"):
                            with T.block("C_reindex_shared"):
                                T.where((ax1_0 + ax1_1) * 32 + ax1_2 < 16)
                                v0 = T.axis.spatial(64, ax0_0_0_ax1_0_0_fused * 16 + ax0)
                                v1 = T.axis.spatial(
                                    128,
                                    ax0_0_1_ax1_0_1_fused * 16 + (ax1_0 * 32 + ax1_1 * 32 + ax1_2),
                                )
                                T.reads(C_reindex_shared[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_reindex_shared[v0, v1]


from tvm import tir


def apply_trace(sch: tir.Schedule) -> None:
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
        pad_value=None,
    )
    sch.transform_layout(
        block=b0,
        buffer=("read", 1),
        index_map=lambda vj, vk: (
            vk,
            vj,
        ),
        pad_value=None,
    )
    sch.transform_layout(
        block=b0,
        buffer=("write", 0),
        index_map=lambda vi, vj: (
            vi,
            vj,
        ),
        pad_value=None,
    )
    sch.transform_block_layout(
        block=b2,
        index_map=lambda vi, vj: (
            vi,
            vj,
        ),
    )
    sch.transform_block_layout(
        block=b3,
        index_map=lambda vi, vk: (
            vi,
            vk,
        ),
    )
    sch.transform_block_layout(
        block=b4,
        index_map=lambda vj, vk: (
            vk,
            vj,
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
    l5, l6, l7 = sch.get_loops(block=b0)
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
        loop=l21, n=5, max_innermost_factor=4, decision=[4, 1, 1, 1, 1]
    )
    l29, l30, l31, l32, l33 = sch.split(
        loop=l21, factors=[v24, v25, v26, v27, v28], preserve_unit_iters=True
    )
    v34, v35, v36, v37, v38 = sch.sample_perfect_tile(
        loop=l22, n=5, max_innermost_factor=4, decision=[1, 8, 1, 1, 1]
    )
    l39, l40, l41, l42, l43 = sch.split(
        loop=l22, factors=[v34, v35, v36, v37, v38], preserve_unit_iters=True
    )
    v44, v45, v46 = sch.sample_perfect_tile(
        loop=l23, n=3, max_innermost_factor=4, decision=[1, 1, 2]
    )
    l47, l48, l49 = sch.split(loop=l23, factors=[v44, v45, v46], preserve_unit_iters=True)
    sch.reorder(l29, l39, l30, l40, l31, l41, l47, l48, l32, l42, l49, l33, l43)
    l50 = sch.fuse(l29, l39, preserve_unit_iters=True)
    sch.bind(loop=l50, thread_axis="blockIdx.y")
    l51 = sch.fuse(l30, l40, preserve_unit_iters=True)
    sch.bind(loop=l51, thread_axis="blockIdx.x")
    l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
    sch.bind(loop=l52, thread_axis="threadIdx.y")
    sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=1)
    sch.annotate(
        block_or_loop=b20, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
    )
    b53 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="shared")
    sch.reverse_compute_at(block=b53, loop=l51, preserve_unit_loops=True, index=-1)
    b54 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(block=b54, loop=l52, preserve_unit_loops=True, index=-1)
    v55 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0
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
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
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
        decision=4,
    )
    sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v135)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch")
    l136, l137, l138, l139 = sch.get_loops(block=b53)
    l140, l141, l142 = sch.split(loop=l139, factors=[None, 1, 32], preserve_unit_iters=True)
    sch.bind(loop=l142, thread_axis="threadIdx.x")
    sch.bind(loop=l141, thread_axis="threadIdx.y")
    sch.unannotate(block_or_loop=b73, ann_key="meta_schedule.cooperative_fetch")
    l143, l144, l145, l146, l147 = sch.get_loops(block=b73)
    l148, l149, l150, l151 = sch.split(
        loop=l147, factors=[None, 1, 32, 2], preserve_unit_iters=True
    )
    sch.vectorize(loop=l151)
    sch.bind(loop=l150, thread_axis="threadIdx.x")
    sch.bind(loop=l149, thread_axis="threadIdx.y")
    sch.unannotate(block_or_loop=b82, ann_key="meta_schedule.cooperative_fetch")
    l152, l153, l154, l155, l156 = sch.get_loops(block=b82)
    l157, l158, l159, l160 = sch.split(
        loop=l156, factors=[None, 1, 32, 8], preserve_unit_iters=True
    )
    sch.vectorize(loop=l160)
    sch.bind(loop=l159, thread_axis="threadIdx.x")
    sch.bind(loop=l158, thread_axis="threadIdx.y")
    b161 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b161, ann_key="meta_schedule.unroll_explicit")
    b162, b163, b164, b165, b166, b167, b168 = sch.get_child_blocks(b161)
    l169, l170, l171, l172, l173, l174, l175, l176 = sch.get_loops(block=b162)
    sch.annotate(block_or_loop=l169, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l169, ann_key="pragma_unroll_explicit", ann_val=1)
    l177, l178, l179, l180, l181, l182, l183, l184 = sch.get_loops(block=b163)
    sch.annotate(block_or_loop=l177, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l177, ann_key="pragma_unroll_explicit", ann_val=1)
    l185, l186, l187, l188, l189, l190, l191 = sch.get_loops(block=b164)
    sch.annotate(block_or_loop=l185, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l185, ann_key="pragma_unroll_explicit", ann_val=1)
    l192, l193, l194, l195, l196, l197, l198 = sch.get_loops(block=b165)
    sch.annotate(block_or_loop=l192, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l192, ann_key="pragma_unroll_explicit", ann_val=1)
    l199, l200, l201, l202, l203, l204, l205, l206, l207, l208 = sch.get_loops(block=b166)
    sch.annotate(block_or_loop=l199, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l199, ann_key="pragma_unroll_explicit", ann_val=1)
    l209, l210, l211, l212, l213 = sch.get_loops(block=b167)
    sch.annotate(block_or_loop=l209, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l209, ann_key="pragma_unroll_explicit", ann_val=1)
    l214, l215, l216, l217, l218, l219 = sch.get_loops(block=b168)
    sch.annotate(block_or_loop=l214, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
    sch.annotate(block_or_loop=l214, ann_key="pragma_unroll_explicit", ann_val=1)
    b220 = sch.get_block(name="C_o", func_name="main")
    l221, l222, l223, l224, l225, l226, l227, l228, l229, l230 = sch.get_loops(block=b220)
    b231 = sch.decompose_reduction(block=b220, loop=l224)
    sch.unannotate(block_or_loop=b231, ann_key="meta_schedule.auto_tensorize")
    sch.annotate(
        block_or_loop=b231, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16"
    )
    sch.unannotate(block_or_loop=b220, ann_key="meta_schedule.auto_tensorize_init")
    sch.unannotate(block_or_loop=b231, ann_key="meta_schedule.auto_tensorize_init")
    b232 = sch.get_block(name="C_o_init", func_name="main")
    sch.unannotate(block_or_loop=b232, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b232, tensor_intrin="wmma_fill_16x16x16_f16")
    b233 = sch.get_block(name="A_reindex_shared_wmma.matrix_a_o", func_name="main")
    sch.unannotate(block_or_loop=b233, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b233, tensor_intrin="wmma_load_16x16x16_f16_a")
    b234 = sch.get_block(name="B_reindex_shared_wmma.matrix_b_o", func_name="main")
    sch.unannotate(block_or_loop=b234, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b234, tensor_intrin="wmma_load_16x16x16_f16_b")
    b235 = sch.get_block(name="C_o_update", func_name="main")
    sch.unannotate(block_or_loop=b235, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b235, tensor_intrin="wmma_sync_16x16x16_f16f16f16")
    b236 = sch.get_block(name="C_reindex_shared_wmma.accumulator_o", func_name="main")
    sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b236, tensor_intrin="wmma_store_16x16x16_f16_shared")


"""
 ID | Name |   FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done
-----------------------------------------------------------------------------------------------------
  0 | main | 524288 |      1 |       234.4426 |       2.2363 |                2.2363 |   1000 |    Y
-----------------------------------------------------------------------------------------------------
Total trials: 1000
Total latency (us): 2.23632

best tuning record is 0, min_run_time is 2.236317443362084e-06
"""
