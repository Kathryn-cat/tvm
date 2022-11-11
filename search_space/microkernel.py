import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class Microkernel_128x128x32:
    @T.prim_func
    def main(
        A: T.Buffer[(128, 32), "float16"],
        B: T.Buffer[(32, 128), "float16"],
        C: T.Buffer[(128, 128), "float16"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
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
        C_reindex_shared = T.alloc_buffer([128, 128], dtype="float16", scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer(
            [128, 128], dtype="float16", scope="wmma.accumulator"
        )
        A_reindex_shared = T.alloc_buffer([128, 32], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([32, 128], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer(
            [128, 32], dtype="float16", scope="wmma.matrix_a"
        )
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer(
            [32, 128], dtype="float16", scope="wmma.matrix_b"
        )
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(
            16,
            thread="blockIdx.y",
            annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1},
        ):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(
                        1, 1, 1, 1
                    ):
                        with T.block("C_o_init"):
                            v0_o = T.axis.spatial(
                                8,
                                ax0_0_4_init
                                + ax0_0_0_ax1_0_0_fused // 4 * 2
                                + ax0_0_1_ax1_0_1_fused
                                + ax0_0_3_init,
                            )
                            v1_o = T.axis.spatial(
                                8,
                                ax1_0_4_init
                                + ax0_0_0_ax1_0_0_fused % 4 * 2
                                + ax0_0_2_ax1_0_2_fused
                                + ax1_0_3_init,
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
                        for ax0_ax1_fused_0 in T.serial(2):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block("A_reindex_shared"):
                                            v0 = T.axis.spatial(
                                                128,
                                                ax0_0_0_ax1_0_0_fused // 4 * 32
                                                + ax0_0_1_ax1_0_1_fused * 16
                                                + (
                                                    ax0_ax1_fused_0 * 256
                                                    + ax0_ax1_fused_1 * 128
                                                    + ax0_ax1_fused_2 * 4
                                                    + ax0_ax1_fused_3
                                                )
                                                // 32,
                                            )
                                            v1 = T.axis.spatial(
                                                32,
                                                (
                                                    ax0_ax1_fused_0 * 256
                                                    + ax0_ax1_fused_1 * 128
                                                    + ax0_ax1_fused_2 * 4
                                                    + ax0_ax1_fused_3
                                                )
                                                % 32,
                                            )
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(16):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    with T.block("B_reindex_shared"):
                                        v0 = T.axis.spatial(
                                            32,
                                            (
                                                ax0_ax1_fused_0 * 64
                                                + ax0_ax1_fused_1 * 32
                                                + ax0_ax1_fused_2
                                            )
                                            // 32,
                                        )
                                        v1 = T.axis.spatial(
                                            128,
                                            ax0_0_0_ax1_0_0_fused % 4 * 32
                                            + (
                                                ax0_ax1_fused_0 * 64
                                                + ax0_ax1_fused_1 * 32
                                                + ax0_ax1_fused_2
                                            )
                                            % 32,
                                        )
                                        T.reads(B[v0, v1])
                                        T.writes(B_reindex_shared[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(1):
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(
                                        8, ax0_0_0_ax1_0_0_fused // 4 * 2 + ax0_0_1_ax1_0_1_fused
                                    )
                                    v1_o = T.axis.spatial(2, ax1_0)
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
                                    v0_o = T.axis.spatial(2, ax0_0)
                                    v1_o = T.axis.spatial(
                                        8, ax0_0_0_ax1_0_0_fused % 4 * 2 + ax0_0_2_ax1_0_2_fused
                                    )
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
                                        8,
                                        ax0_0_4
                                        + ax0_0_0_ax1_0_0_fused // 4 * 2
                                        + ax0_0_1_ax1_0_1_fused
                                        + ax0_0_3,
                                    )
                                    v1_o = T.axis.spatial(
                                        8,
                                        ax1_0_4
                                        + ax0_0_0_ax1_0_0_fused % 4 * 2
                                        + ax0_0_2_ax1_0_2_fused
                                        + ax1_0_3,
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
                            v0_o = T.axis.spatial(
                                8, ax0_0_0_ax1_0_0_fused // 4 * 2 + ax0_0_1_ax1_0_1_fused
                            )
                            v1_o = T.axis.spatial(
                                8, ax0_0_0_ax1_0_0_fused % 4 * 2 + ax0_0_2_ax1_0_2_fused
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
                    for ax1_1 in T.thread_binding(2, thread="threadIdx.y"):
                        for ax1_2 in T.thread_binding(32, thread="threadIdx.x"):
                            with T.block("C_reindex_shared"):
                                T.where((ax1_0 * 2 + ax1_1) * 32 + ax1_2 < 32)
                                v0 = T.axis.spatial(
                                    128,
                                    ax0_0_0_ax1_0_0_fused // 4 * 32
                                    + ax0_0_1_ax1_0_1_fused * 16
                                    + ax0,
                                )
                                v1 = T.axis.spatial(
                                    128,
                                    ax0_0_0_ax1_0_0_fused % 4 * 32
                                    + (ax1_0 * 64 + ax1_1 * 32 + ax1_2),
                                )
                                T.reads(C_reindex_shared[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_reindex_shared[v0, v1]
