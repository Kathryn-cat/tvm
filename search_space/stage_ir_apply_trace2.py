import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class StagedModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "StagedModule", "tir.noalias": True})
        a0 = T.var("int32")
        a1 = T.var("int32")
        b0 = T.var("int32")
        b1 = T.var("int32")
        c0 = T.var("int32")
        c1 = T.var("int32")
        m = T.var("int32")
        n = T.var("int32")
        p = T.var("int32")
        A = T.match_buffer(a, [m, n], dtype="float16")
        B = T.match_buffer(b, [n, p], dtype="float16")
        C = T.match_buffer(c, [m, p], dtype="float16")
        # body
        # with T.block("root")
        A_pad = T.alloc_buffer([(m + 127) // 128 * 128, (n + 31) // 32 * 32], dtype="float16")
        B_pad = T.alloc_buffer([(n + 31) // 32 * 32, (p + 127) // 128 * 128], dtype="float16")
        C_pad = T.alloc_buffer([(m + 127) // 128 * 128, (p + 127) // 128 * 128], dtype="float16")
        for i, j in T.grid((m + 127) // 128 * 128, (n + 31) // 32 * 32):
            with T.block("A_pad"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(A_pad[vi, vj])
                A_pad[vi, vj] = T.if_then_else(
                    vi < m and vj < n, A[vi, vj], T.float16(0), dtype="float16"
                )
        for i, j in T.grid((n + 31) // 32 * 32, (p + 127) // 128 * 128):
            with T.block("B_pad"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(B_pad[vi, vj])
                B_pad[vi, vj] = T.if_then_else(
                    vi < n and vj < p, B[vi, vj], T.float16(0), dtype="float16"
                )
        for i_0 in T.thread_binding((m + 127) // 128, thread="blockIdx.y"):
            for j_0 in T.thread_binding((p + 127) // 128, thread="blockIdx.x"):
                with T.block("C_o"):
                    vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                    vk_o = T.axis.reduce(1, 0)
                    T.reads(
                        A_pad[vi_o * 128 : vi_o * 128 + 128, 0 : (n + 31) // 32 * 32],
                        B_pad[0 : (n + 31) // 32 * 32, vj_o * 128 : vj_o * 128 + 128],
                    )
                    T.writes(C_pad[vi_o * 128 : vi_o * 128 + 128, vj_o * 128 : vj_o * 128 + 128])
                    with T.init():
                        for i_1, j_1 in T.grid(128, 128):
                            with T.block("C_init"):
                                vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                                T.reads()
                                T.writes(C_pad[vi_o * 128 + vi_i_init, vj_o * 128 + vj_i_init])
                                C_pad[vi_o * 128 + vi_i_init, vj_o * 128 + vj_i_init] = T.float16(0)
                    for k_0 in T.serial((n + 31) // 32):
                        with T.block("C_o"):
                            vi_i_o = T.axis.spatial(1, 0)
                            vj_i_o = T.axis.spatial(1, 0)
                            vk_i_o = T.axis.reduce((n + 31) // 32, k_0)
                            T.reads(
                                C_pad[vi_o * 128 : vi_o * 128 + 128, vj_o * 128 : vj_o * 128 + 128],
                                A_pad[
                                    vi_o * 128 : vi_o * 128 + 128, vk_i_o * 32 : vk_i_o * 32 + 32
                                ],
                                B_pad[
                                    vk_i_o * 32 : vk_i_o * 32 + 32, vj_o * 128 : vj_o * 128 + 128
                                ],
                            )
                            T.writes(
                                C_pad[vi_o * 128 : vi_o * 128 + 128, vj_o * 128 : vj_o * 128 + 128]
                            )
                            A_pad_shared = T.alloc_buffer(
                                [(m + 127) // 128 * 128, (n + 31) // 32 * 32],
                                dtype="float16",
                                scope="shared",
                            )
                            B_pad_shared = T.alloc_buffer(
                                [(n + 31) // 32 * 32, (p + 127) // 128 * 128],
                                dtype="float16",
                                scope="shared",
                            )
                            A_pad_shared_wmma_matrix_a = T.alloc_buffer(
                                [(m + 127) // 128 * 128, (n + 31) // 32 * 32],
                                dtype="float16",
                                scope="wmma.matrix_a",
                            )
                            B_pad_shared_wmma_matrix_b = T.alloc_buffer(
                                [(n + 31) // 32 * 32, (p + 127) // 128 * 128],
                                dtype="float16",
                                scope="wmma.matrix_b",
                            )
                            C_pad_shared = T.alloc_buffer(
                                [(m + 127) // 128 * 128, (p + 127) // 128 * 128],
                                dtype="float16",
                                scope="shared",
                            )
                            C_pad_shared_wmma_accumulator = T.alloc_buffer(
                                [(m + 127) // 128 * 128, (p + 127) // 128 * 128],
                                dtype="float16",
                                scope="wmma.accumulator",
                            )
                            for ax0_ax1_fused_0 in T.serial(8):
                                for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(
                                        32, thread="threadIdx.x"
                                    ):
                                        for ax0_ax1_fused_3 in T.vectorized(4):
                                            with T.block("B_pad_shared"):
                                                v0 = T.axis.spatial(
                                                    (n + 31) // 32 * 32,
                                                    (
                                                        ax0_ax1_fused_0 * 512
                                                        + ax0_ax1_fused_1 * 128
                                                        + ax0_ax1_fused_2 * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    // 128
                                                    + vk_i_o * 32,
                                                )
                                                v1 = T.axis.spatial(
                                                    (p + 127) // 128 * 128,
                                                    (
                                                        ax0_ax1_fused_0 * 512
                                                        + ax0_ax1_fused_1 * 128
                                                        + ax0_ax1_fused_2 * 4
                                                        + ax0_ax1_fused_3
                                                    )
                                                    % 128
                                                    + vj_o * 128,
                                                )
                                                T.reads(B_pad[v0, v1])
                                                T.writes(B_pad_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                B_pad_shared[v0, v1] = B_pad[v0, v1]
                            for ax0_ax1_fused_0 in T.serial(16):
                                for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(
                                        32, thread="threadIdx.x"
                                    ):
                                        for ax0_ax1_fused_3 in T.vectorized(2):
                                            with T.block("A_pad_shared"):
                                                v0 = T.axis.spatial(
                                                    (m + 127) // 128 * 128,
                                                    (
                                                        ax0_ax1_fused_0 * 256
                                                        + ax0_ax1_fused_1 * 64
                                                        + ax0_ax1_fused_2 * 2
                                                        + ax0_ax1_fused_3
                                                    )
                                                    // 32
                                                    + vi_o * 128,
                                                )
                                                v1 = T.axis.spatial(
                                                    (n + 31) // 32 * 32,
                                                    (
                                                        ax0_ax1_fused_0 * 256
                                                        + ax0_ax1_fused_1 * 64
                                                        + ax0_ax1_fused_2 * 2
                                                        + ax0_ax1_fused_3
                                                    )
                                                    % 32
                                                    + vk_i_o * 32,
                                                )
                                                T.reads(A_pad[v0, v1])
                                                T.writes(A_pad_shared[v0, v1])
                                                T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                                A_pad_shared[v0, v1] = A_pad[v0, v1]
                            for ax0_0, ax1_0 in T.grid(8, 2):
                                with T.block("A_pad_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial((m + 127) // 128 * 8, vi_o * 8 + ax0_0)
                                    v1_o = T.axis.spatial((n + 31) // 32 * 2, vk_i_o * 2 + ax1_0)
                                    T.reads(
                                        A_pad_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.writes(
                                        A_pad_shared_wmma_matrix_a[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("A_pad_shared_wmma.matrix_a"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(
                                                A_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                            )
                                            T.writes(
                                                A_pad_shared_wmma_matrix_a[
                                                    v0_o * 16 + v0_i, v1_o * 16 + v1_i
                                                ]
                                            )
                                            A_pad_shared_wmma_matrix_a[
                                                v0_o * 16 + v0_i, v1_o * 16 + v1_i
                                            ] = A_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for ax0_0, ax1_0 in T.grid(2, 8):
                                with T.block("B_pad_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial((n + 31) // 32 * 2, ax0_0 + vk_i_o * 2)
                                    v1_o = T.axis.spatial((p + 127) // 128 * 8, ax1_0 + vj_o * 8)
                                    T.reads(
                                        B_pad_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.writes(
                                        B_pad_shared_wmma_matrix_b[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("B_pad_shared_wmma.matrix_b"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(
                                                B_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                            )
                                            T.writes(
                                                B_pad_shared_wmma_matrix_b[
                                                    v0_o * 16 + v0_i, v1_o * 16 + v1_i
                                                ]
                                            )
                                            B_pad_shared_wmma_matrix_b[
                                                v0_o * 16 + v0_i, v1_o * 16 + v1_i
                                            ] = B_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                            for i_1_0_0_j_1_0_0_fused in T.thread_binding(4, thread="threadIdx.y"):
                                for i_1_0_1, j_1_0_1, k_1_0 in T.grid(4, 4, 2):
                                    with T.block("C_o"):
                                        vi_i_i_o = T.axis.spatial(
                                            8, i_1_0_0_j_1_0_0_fused // 2 * 4 + i_1_0_1
                                        )
                                        vj_i_i_o = T.axis.spatial(
                                            8, i_1_0_0_j_1_0_0_fused % 2 * 4 + j_1_0_1
                                        )
                                        vk_i_i_o = T.axis.reduce(2, k_1_0)
                                        T.reads(
                                            C_pad_shared_wmma_accumulator[
                                                vi_o * 128
                                                + vi_i_i_o * 16 : vi_o * 128
                                                + vi_i_i_o * 16
                                                + 16,
                                                vj_o * 128
                                                + vj_i_i_o * 16 : vj_o * 128
                                                + vj_i_i_o * 16
                                                + 16,
                                            ],
                                            A_pad_shared_wmma_matrix_a[
                                                vi_o * 128
                                                + vi_i_i_o * 16 : vi_o * 128
                                                + vi_i_i_o * 16
                                                + 16,
                                                vk_i_o * 32
                                                + vk_i_i_o * 16 : vk_i_o * 32
                                                + vk_i_i_o * 16
                                                + 16,
                                            ],
                                            B_pad_shared_wmma_matrix_b[
                                                vk_i_o * 32
                                                + vk_i_i_o * 16 : vk_i_o * 32
                                                + vk_i_i_o * 16
                                                + 16,
                                                vj_o * 128
                                                + vj_i_i_o * 16 : vj_o * 128
                                                + vj_i_i_o * 16
                                                + 16,
                                            ],
                                        )
                                        T.writes(
                                            C_pad_shared_wmma_accumulator[
                                                vi_o * 128
                                                + vi_i_i_o * 16 : vi_o * 128
                                                + vi_i_i_o * 16
                                                + 16,
                                                vj_o * 128
                                                + vj_i_i_o * 16 : vj_o * 128
                                                + vj_i_i_o * 16
                                                + 16,
                                            ]
                                        )
                                        A_1 = T.match_buffer(
                                            A_pad_shared_wmma_matrix_a[
                                                vi_o * 128
                                                + vi_i_i_o * 16 : vi_o * 128
                                                + vi_i_i_o * 16
                                                + 16,
                                                vk_i_o * 32
                                                + vk_i_i_o * 16 : vk_i_o * 32
                                                + vk_i_i_o * 16
                                                + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            strides=[a1, a0],
                                            scope="wmma.matrix_a",
                                            offset_factor=16,
                                        )
                                        B_1 = T.match_buffer(
                                            B_pad_shared_wmma_matrix_b[
                                                vk_i_o * 32
                                                + vk_i_i_o * 16 : vk_i_o * 32
                                                + vk_i_i_o * 16
                                                + 16,
                                                vj_o * 128
                                                + vj_i_i_o * 16 : vj_o * 128
                                                + vj_i_i_o * 16
                                                + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            strides=[b1, b0],
                                            scope="wmma.matrix_b",
                                            offset_factor=16,
                                        )
                                        C_1 = T.match_buffer(
                                            C_pad_shared_wmma_accumulator[
                                                vi_o * 128
                                                + vi_i_i_o * 16 : vi_o * 128
                                                + vi_i_i_o * 16
                                                + 16,
                                                vj_o * 128
                                                + vj_i_i_o * 16 : vj_o * 128
                                                + vj_i_i_o * 16
                                                + 16,
                                            ],
                                            [16, 16],
                                            dtype="float16",
                                            strides=[c1, c0],
                                            scope="wmma.accumulator",
                                            offset_factor=16,
                                        )
                                        T.evaluate(
                                            T.tvm_mma_sync(
                                                C_1.data,
                                                C_1.elem_offset // c1 // 16 * (c1 // 16)
                                                + C_1.elem_offset % c1 // 16,
                                                A_1.data,
                                                A_1.elem_offset // a1 // 16 * (a1 // 16)
                                                + A_1.elem_offset % a1 // 16,
                                                B_1.data,
                                                B_1.elem_offset // b1 // 16 * (b1 // 16)
                                                + B_1.elem_offset % b1 // 16,
                                                C_1.data,
                                                C_1.elem_offset // c1 // 16 * (c1 // 16)
                                                + C_1.elem_offset % c1 // 16,
                                                dtype="handle",
                                            )
                                        )
                            for ax0_0, ax1_0 in T.grid(8, 8):
                                with T.block("C_pad_shared_wmma.accumulator_o"):
                                    v0_o = T.axis.spatial((m + 127) // 128 * 8, ax0_0 + vi_o * 8)
                                    v1_o = T.axis.spatial((p + 127) // 128 * 8, ax1_0 + vj_o * 8)
                                    T.reads(
                                        C_pad_shared_wmma_accumulator[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    T.writes(
                                        C_pad_shared[
                                            v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                        ]
                                    )
                                    for ax0_1, ax1_1 in T.grid(16, 16):
                                        with T.block("C_pad_shared_wmma.accumulator"):
                                            v0_i, v1_i = T.axis.remap("SS", [ax0_1, ax1_1])
                                            T.reads(
                                                C_pad_shared_wmma_accumulator[
                                                    v0_o * 16 + v0_i, v1_o * 16 + v1_i
                                                ]
                                            )
                                            T.writes(
                                                C_pad_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                            )
                                            C_pad_shared[
                                                v0_o * 16 + v0_i, v1_o * 16 + v1_i
                                            ] = C_pad_shared_wmma_accumulator[
                                                v0_o * 16 + v0_i, v1_o * 16 + v1_i
                                            ]
                            for ax0_ax1_fused_0 in T.serial(128):
                                for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(
                                        32, thread="threadIdx.x"
                                    ):
                                        for ax0_ax1_fused_3 in T.vectorized(1):
                                            with T.block("C_pad_shared"):
                                                v0 = T.axis.spatial(
                                                    (m + 127) // 128 * 128,
                                                    (
                                                        ax0_ax1_fused_3
                                                        + ax0_ax1_fused_0 * 128
                                                        + ax0_ax1_fused_1 * 32
                                                        + ax0_ax1_fused_2
                                                    )
                                                    // 128
                                                    + vi_o * 128,
                                                )
                                                v1 = T.axis.spatial(
                                                    (p + 127) // 128 * 128,
                                                    (
                                                        ax0_ax1_fused_3
                                                        + ax0_ax1_fused_0 * 128
                                                        + ax0_ax1_fused_1 * 32
                                                        + ax0_ax1_fused_2
                                                    )
                                                    % 128
                                                    + vj_o * 128,
                                                )
                                                T.reads(C_pad_shared[v0, v1])
                                                T.writes(C_pad[v0, v1])
                                                C_pad[v0, v1] = C_pad_shared[v0, v1]
        for i, j in T.grid(m, p):
            with T.block("C_pad"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(C_pad[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = C_pad[vi, vj]
