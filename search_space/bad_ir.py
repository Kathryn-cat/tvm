import tvm
from tvm.script import tir as T


@tvm.script.ir_module
class BadModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "BadModule", "tir.noalias": True})
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
                            for ax0, ax1 in T.grid(32, 128):
                                with T.block("B_pad_shared"):
                                    v0 = T.axis.spatial((n + 31) // 32 * 32, vk_i_o * 32 + ax0)
                                    v1 = T.axis.spatial((p + 127) // 128 * 128, vj_o * 128 + ax1)
                                    T.reads(B_pad[v0, v1])
                                    T.writes(B_pad_shared[v0, v1])
                                    B_pad_shared[v0, v1] = B_pad[v0, v1]
                            for ax0, ax1 in T.grid(128, 32):
                                with T.block("A_pad_shared"):
                                    v0 = T.axis.spatial((m + 127) // 128 * 128, vi_o * 128 + ax0)
                                    v1 = T.axis.spatial((n + 31) // 32 * 32, vk_i_o * 32 + ax1)
                                    T.reads(A_pad[v0, v1])
                                    T.writes(A_pad_shared[v0, v1])
                                    A_pad_shared[v0, v1] = A_pad[v0, v1]
                            for ax0_0, ax1_0 in T.grid(8, 2):
                                with T.block("A_pad_shared_wmma.matrix_a_o"):
                                    v0_o, v1_o = T.axis.remap("SS", [ax0_0, ax1_0])
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
                                            v0_i = T.axis.spatial(16, ax0_1 + vi_o * 128)
                                            v1_i = T.axis.spatial(16, ax1_1 + vk_i_o * 32)
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
                            for ax0, ax1 in T.grid(32, 128):
                                with T.block("B_pad_shared_wmma.matrix_b"):
                                    v0 = T.axis.spatial((n + 31) // 32 * 32, vk_i_o * 32 + ax0)
                                    v1 = T.axis.spatial((p + 127) // 128 * 128, vj_o * 128 + ax1)
                                    T.reads(B_pad_shared[v0, v1])
                                    T.writes(B_pad_shared_wmma_matrix_b[v0, v1])
                                    B_pad_shared_wmma_matrix_b[v0, v1] = B_pad_shared[v0, v1]
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
                                        for i_1_1, j_1_1, k_1_1 in T.grid(16, 16, 16):
                                            with T.block("C"):
                                                vi_i_i_i, vj_i_i_i, vk_i_i_i = T.axis.remap(
                                                    "SSR", [i_1_1, j_1_1, k_1_1]
                                                )
                                                T.reads(
                                                    C_pad_shared_wmma_accumulator[
                                                        vi_o * 128 + vi_i_i_o * 16 + vi_i_i_i,
                                                        vj_o * 128 + vj_i_i_o * 16 + vj_i_i_i,
                                                    ],
                                                    A_pad_shared_wmma_matrix_a[
                                                        vi_o * 128 + vi_i_i_o * 16 + vi_i_i_i,
                                                        vk_i_o * 32 + vk_i_i_o * 16 + vk_i_i_i,
                                                    ],
                                                    B_pad_shared_wmma_matrix_b[
                                                        vk_i_o * 32 + vk_i_i_o * 16 + vk_i_i_i,
                                                        vj_o * 128 + vj_i_i_o * 16 + vj_i_i_i,
                                                    ],
                                                )
                                                T.writes(
                                                    C_pad_shared_wmma_accumulator[
                                                        vi_o * 128 + vi_i_i_o * 16 + vi_i_i_i,
                                                        vj_o * 128 + vj_i_i_o * 16 + vj_i_i_i,
                                                    ]
                                                )
                                                C_pad_shared_wmma_accumulator[
                                                    vi_o * 128 + vi_i_i_o * 16 + vi_i_i_i,
                                                    vj_o * 128 + vj_i_i_o * 16 + vj_i_i_i,
                                                ] = (
                                                    C_pad_shared_wmma_accumulator[
                                                        vi_o * 128 + vi_i_i_o * 16 + vi_i_i_i,
                                                        vj_o * 128 + vj_i_i_o * 16 + vj_i_i_i,
                                                    ]
                                                    + A_pad_shared_wmma_matrix_a[
                                                        vi_o * 128 + vi_i_i_o * 16 + vi_i_i_i,
                                                        vk_i_o * 32 + vk_i_i_o * 16 + vk_i_i_i,
                                                    ]
                                                    * B_pad_shared_wmma_matrix_b[
                                                        vk_i_o * 32 + vk_i_i_o * 16 + vk_i_i_i,
                                                        vj_o * 128 + vj_i_i_o * 16 + vj_i_i_i,
                                                    ]
                                                )
                            for ax0, ax1 in T.grid(128, 128):
                                with T.block("C_pad_shared_wmma.accumulator"):
                                    v0 = T.axis.spatial((m + 127) // 128 * 128, vi_o * 128 + ax0)
                                    v1 = T.axis.spatial((p + 127) // 128 * 128, vj_o * 128 + ax1)
                                    T.reads(C_pad_shared_wmma_accumulator[v0, v1])
                                    T.writes(C_pad_shared[v0, v1])
                                    C_pad_shared[v0, v1] = C_pad_shared_wmma_accumulator[v0, v1]
                            for ax0, ax1 in T.grid(128, 128):
                                with T.block("C_pad_shared"):
                                    v0 = T.axis.spatial((m + 127) // 128 * 128, vi_o * 128 + ax0)
                                    v1 = T.axis.spatial((p + 127) // 128 * 128, vj_o * 128 + ax1)
                                    T.reads(C_pad_shared[v0, v1])
                                    T.writes(C_pad[v0, v1])
                                    C_pad[v0, v1] = C_pad_shared[v0, v1]
        for i, j in T.grid(m, p):
            with T.block("C_pad"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(C_pad[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = C_pad[vi, vj]
