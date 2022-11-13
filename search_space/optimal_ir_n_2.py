# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer[(1024, 1024), "float16"],
        B: T.Buffer[(1024, 1024), "float16"],
        C: T.Buffer[(1024, 1024), "float16"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "staticMatmul", "tir.noalias": True})
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
        C_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
        C_shared_wmma_accumulator = T.alloc_buffer(
            [1024, 1024], dtype="float16", scope="wmma.accumulator"
        )
        A_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
        B_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
        A_shared_wmma_matrix_a = T.alloc_buffer(
            [1024, 1024], dtype="float16", scope="wmma.matrix_a"
        )
        B_shared_wmma_matrix_b = T.alloc_buffer(
            [1024, 1024], dtype="float16", scope="wmma.matrix_b"
        )
        for i_0_0_j_0_0_fused in T.thread_binding(512, thread="blockIdx.y"):
            for i_0_1_j_0_1_fused in T.thread_binding(4, thread="blockIdx.x"):
                for i_0_2_j_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                    for i_0_3_init, j_0_3_init, i_0_4_init, j_0_4_init in T.grid(1, 1, 1, 1):
                        with T.block("C_o_init"):
                            vi_o = T.axis.spatial(
                                64, i_0_3_init + i_0_4_init + i_0_0_j_0_0_fused // 8
                            )
                            vj_o = T.axis.spatial(
                                64,
                                j_0_4_init
                                + i_0_0_j_0_0_fused % 8 * 8
                                + i_0_1_j_0_1_fused * 2
                                + i_0_2_j_0_2_fused
                                + j_0_3_init,
                            )
                            T.reads()
                            T.writes(
                                C_shared_wmma_accumulator[
                                    vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16
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
                                C_shared_wmma_accumulator[
                                    vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16
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
                    for k_0_0 in T.serial(32):
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(2):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(
                                                1024,
                                                i_0_0_j_0_0_fused // 8 * 16
                                                + (
                                                    ax0_ax1_fused_0 * 128
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                // 32,
                                            )
                                            v1 = T.axis.spatial(
                                                1024,
                                                k_0_0 * 32
                                                + (
                                                    ax0_ax1_fused_0 * 128
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                % 32,
                                            )
                                            T.reads(A[v0, v1])
                                            T.writes(A_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(8):
                            for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(2):
                                        with T.block("B_shared"):
                                            v0 = T.axis.spatial(
                                                1024,
                                                k_0_0 * 32
                                                + (
                                                    ax0_ax1_fused_0 * 128
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                // 32,
                                            )
                                            v1 = T.axis.spatial(
                                                1024,
                                                i_0_0_j_0_0_fused % 8 * 128
                                                + i_0_1_j_0_1_fused * 32
                                                + (
                                                    ax0_ax1_fused_0 * 128
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                % 32,
                                            )
                                            T.reads(B[v0, v1])
                                            T.writes(B_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            B_shared[v0, v1] = B[v0, v1]
                        for k_0_1 in T.serial(1):
                            for ax0_0, ax1_0 in T.grid(1, 2):
                                with T.block("A_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(64, i_0_0_j_0_0_fused // 8)
                                    v1_o = T.axis.spatial(64, k_0_0 * 2 + ax1_0)
                                    T.reads(
                                        A_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.writes(
                                        A_shared_wmma_matrix_a[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    A_1 = T.match_buffer(
                                        A_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[s1, s0],
                                        scope="shared",
                                        offset_factor=16,
                                    )
                                    C_2 = T.match_buffer(
                                        A_shared_wmma_matrix_a[
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
                                with T.block("B_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(64, k_0_0 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(
                                        64,
                                        i_0_0_j_0_0_fused % 8 * 8
                                        + i_0_1_j_0_1_fused * 2
                                        + i_0_2_j_0_2_fused,
                                    )
                                    T.reads(
                                        B_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.writes(
                                        B_shared_wmma_matrix_b[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    A_2 = T.match_buffer(
                                        B_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[s1_1, s0_1],
                                        scope="shared",
                                        offset_factor=16,
                                    )
                                    C_3 = T.match_buffer(
                                        B_shared_wmma_matrix_b[
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
                            for i_0_3, j_0_3, k_0_2, i_0_4, j_0_4 in T.grid(1, 1, 2, 1, 1):
                                with T.block("C_o_update"):
                                    vi_o = T.axis.spatial(
                                        64, i_0_3 + i_0_4 + i_0_0_j_0_0_fused // 8
                                    )
                                    vj_o = T.axis.spatial(
                                        64,
                                        j_0_4
                                        + i_0_0_j_0_0_fused % 8 * 8
                                        + i_0_1_j_0_1_fused * 2
                                        + i_0_2_j_0_2_fused
                                        + j_0_3,
                                    )
                                    vk_o = T.axis.reduce(64, k_0_0 * 2 + k_0_1 * 2 + k_0_2)
                                    T.reads(
                                        C_shared_wmma_accumulator[
                                            vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16
                                        ],
                                        A_shared_wmma_matrix_a[
                                            vi_o * 16 : vi_o * 16 + 16, vk_o * 16 : vk_o * 16 + 16
                                        ],
                                        B_shared_wmma_matrix_b[
                                            vk_o * 16 : vk_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16
                                        ],
                                    )
                                    T.writes(
                                        C_shared_wmma_accumulator[
                                            vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16
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
                                        A_shared_wmma_matrix_a[
                                            vi_o * 16 : vi_o * 16 + 16, vk_o * 16 : vk_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[a1, a0],
                                        scope="wmma.matrix_a",
                                        offset_factor=16,
                                    )
                                    B_1 = T.match_buffer(
                                        B_shared_wmma_matrix_b[
                                            vk_o * 16 : vk_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16
                                        ],
                                        [16, 16],
                                        dtype="float16",
                                        strides=[b1, b0],
                                        scope="wmma.matrix_b",
                                        offset_factor=16,
                                    )
                                    C_4 = T.match_buffer(
                                        C_shared_wmma_accumulator[
                                            vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16
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
                        with T.block("C_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(64, i_0_0_j_0_0_fused // 8)
                            v1_o = T.axis.spatial(
                                64,
                                i_0_0_j_0_0_fused % 8 * 8
                                + i_0_1_j_0_1_fused * 2
                                + i_0_2_j_0_2_fused,
                            )
                            T.reads(
                                C_shared_wmma_accumulator[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ]
                            )
                            T.writes(
                                C_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16]
                            )
                            A_4 = T.match_buffer(
                                C_shared_wmma_accumulator[
                                    v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                ],
                                [16, 16],
                                dtype="float16",
                                strides=[d1_3, d0_3],
                                scope="wmma.accumulator",
                                offset_factor=16,
                            )
                            C_5 = T.match_buffer(
                                C_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16],
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
                            with T.block("C_shared"):
                                T.where((ax1_0 * 2 + ax1_1) * 32 + ax1_2 < 32)
                                v0 = T.axis.spatial(1024, i_0_0_j_0_0_fused // 8 * 16 + ax0)
                                v1 = T.axis.spatial(
                                    1024,
                                    i_0_0_j_0_0_fused % 8 * 128
                                    + i_0_1_j_0_1_fused * 32
                                    + (ax1_0 * 64 + ax1_1 * 32 + ax1_2),
                                )
                                T.reads(C_shared[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_shared[v0, v1]
