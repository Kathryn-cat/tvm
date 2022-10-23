import numpy as np
import torch

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _


# from tvm.script import tir as T
@tvm.script.ir_module
class DynModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        s0 = T.var("int32")
        s0_1 = T.var("int32")
        s0_2 = T.var("int32")
        s1 = T.var("int32")
        s1_1 = T.var("int32")
        s1_2 = T.var("int32")
        A = T.match_buffer(a, [1024 * m, 1024 * n], dtype="float16")
        B = T.match_buffer(b, [1024 * n, 1024 * p], dtype="float16")
        C = T.match_buffer(c, [1024 * m, 1024 * p], dtype="float16")
        # body
        # with T.block("root")
        C_shared = T.alloc_buffer([1024 * m, 1024 * p], dtype="float16", scope="shared")
        C_shared_wmma_accumulator = T.alloc_buffer([1024 * m, 1024 * p], dtype="float16", scope="wmma.accumulator")
        A_shared = T.alloc_buffer([1024 * m, 1024 * n], dtype="float16", scope="shared")
        B_shared = T.alloc_buffer([1024 * n, 1024 * p], dtype="float16", scope="shared")
        A_shared_wmma_matrix_a = T.alloc_buffer([1024 * m, 1024 * n], dtype="float16", scope="wmma.matrix_a")
        B_shared_wmma_matrix_b = T.alloc_buffer([1024 * n, 1024 * p], dtype="float16", scope="wmma.matrix_b")
        for i_0_0 in T.thread_binding(m * 16, thread="blockIdx.y"):
            for j_0_0 in T.thread_binding(n * 8, thread="blockIdx.x"):
                for i_0_1_0_j_0_1_0_fused in T.thread_binding(4, thread="threadIdx.y"):
                    for i_0_1_1_init, j_0_1_1_init, i_0_1_2_init, j_0_1_2_init in T.grid(1, 2, 2, 2):
                        with T.block("C_o_init"):
                            vi_o = T.axis.spatial(m * 64, i_0_0 * 4 + i_0_1_0_j_0_1_0_fused // 2 * 2 + i_0_1_1_init * 2 + i_0_1_2_init)
                            vj_o = T.axis.spatial(n * 64, j_0_0 * 8 + i_0_1_0_j_0_1_0_fused % 2 * 4 + j_0_1_1_init * 2 + j_0_1_2_init)
                            T.reads()
                            T.writes(C_shared_wmma_accumulator[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                            C_1 = T.match_buffer(C_shared_wmma_accumulator[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.accumulator", offset_factor=16)
                            T.evaluate(T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // 256 + C_1.elem_offset % 256 // 16, T.float32(0), dtype="handle"))
                    for k_0_0 in T.serial(p * 16):
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(8):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(1024 * m, i_0_0 * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) // 64)
                                            v1 = T.axis.spatial(1024 * n, k_0_0 * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) % 64)
                                            T.reads(A[v0, v1])
                                            T.writes(A_shared[v0, v1])
                                            A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(16):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block("B_shared"):
                                            v0 = T.axis.spatial(1024 * n, k_0_0 * 64 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 128)
                                            v1 = T.axis.spatial(1024 * p, j_0_0 * 128 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 128)
                                            T.reads(B[v0, v1])
                                            T.writes(B_shared[v0, v1])
                                            B_shared[v0, v1] = B[v0, v1]
                        for k_0_1_0 in T.serial(4):
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("A_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(4 * (m * 16), i_0_0 * 4 + i_0_1_0_j_0_1_0_fused // 2 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(4 * (p * 16), k_0_0 * 4 + k_0_1_0)
                                    T.reads(A_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    A_1 = T.match_buffer(A_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[s1, s0], scope="shared", offset_factor=16)
                                    C_2 = T.match_buffer(A_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                    T.evaluate(T.tvm_load_matrix_sync(C_2.data, 16, 16, 16, C_2.elem_offset // 256 + C_2.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_1.data, A_1.elem_offset, s1 * 16, 1, dtype="handle"), s1, "row_major", dtype="handle"))
                            for ax0_0, ax1_0 in T.grid(1, 4):
                                with T.block("B_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(4 * (p * 16), k_0_0 * 4 + k_0_1_0)
                                    v1_o = T.axis.spatial(8 * (n * 8), j_0_0 * 8 + i_0_1_0_j_0_1_0_fused % 2 * 4 + ax1_0)
                                    T.reads(B_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    A_2 = T.match_buffer(B_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[s1_1, s0_1], scope="shared", offset_factor=16)
                                    C_3 = T.match_buffer(B_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                    T.evaluate(T.tvm_load_matrix_sync(C_3.data, 16, 16, 16, C_3.elem_offset // 256 + C_3.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_2.data, A_2.elem_offset, s1_1 * 16, 1, dtype="handle"), s1_1, "row_major", dtype="handle"))
                            for i_0_1_1, j_0_1_1, k_0_1_1, i_0_1_2, j_0_1_2 in T.grid(1, 2, 1, 2, 2):
                                with T.block("C_o_update"):
                                    vi_o = T.axis.spatial(m * 64, i_0_0 * 4 + i_0_1_0_j_0_1_0_fused // 2 * 2 + i_0_1_1 * 2 + i_0_1_2)
                                    vj_o = T.axis.spatial(n * 64, j_0_0 * 8 + i_0_1_0_j_0_1_0_fused % 2 * 4 + j_0_1_1 * 2 + j_0_1_2)
                                    vk_o = T.axis.reduce(p * 64, k_0_0 * 4 + (k_0_1_0 + k_0_1_1))
                                    T.reads(C_shared_wmma_accumulator[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16], A_shared_wmma_matrix_a[vi_o * 16 : vi_o * 16 + 16, vk_o * 16 : vk_o * 16 + 16], B_shared_wmma_matrix_b[vk_o * 16 : vk_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                                    T.writes(C_shared_wmma_accumulator[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                                    A_3 = T.match_buffer(A_shared_wmma_matrix_a[vi_o * 16 : vi_o * 16 + 16, vk_o * 16 : vk_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_shared_wmma_matrix_b[vk_o * 16 : vk_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                    C_4 = T.match_buffer(C_shared_wmma_accumulator[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.accumulator", offset_factor=16)
                                    T.evaluate(T.tvm_mma_sync(C_4.data, C_4.elem_offset // 256 + C_4.elem_offset % 256 // 16, A_3.data, A_3.elem_offset // 256 + A_3.elem_offset % 256 // 16, B_1.data, B_1.elem_offset // 256 + B_1.elem_offset % 256 // 16, C_4.data, C_4.elem_offset // 256 + C_4.elem_offset % 256 // 16, dtype="handle"))
                    for ax0_0, ax1_0 in T.grid(2, 4):
                        with T.block("C_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(4 * (m * 16), i_0_0 * 4 + i_0_1_0_j_0_1_0_fused // 2 * 2 + ax0_0)
                            v1_o = T.axis.spatial(8 * (n * 8), j_0_0 * 8 + i_0_1_0_j_0_1_0_fused % 2 * 4 + ax1_0)
                            T.reads(C_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(C_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            A_4 = T.match_buffer(C_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.accumulator", offset_factor=16)
                            C_5 = T.match_buffer(C_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[s1_2, s0_2], scope="shared", offset_factor=16)
                            T.evaluate(T.tvm_store_matrix_sync(A_4.data, 16, 16, 16, A_4.elem_offset // 256 + A_4.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_5.data, C_5.elem_offset, s1_2 * 16, 2, dtype="handle"), s1_2, "row_major", dtype="handle"))
                for ax0, ax1_0 in T.grid(64, 1):
                    for ax1_1 in T.thread_binding(4, thread="threadIdx.y"):
                        for ax1_2 in T.thread_binding(32, thread="threadIdx.x"):
                            for ax1_3 in T.vectorized(4):
                                with T.block("C_shared"):
                                    T.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 4 + ax1_3 < 128)
                                    v0 = T.axis.spatial(1024 * m, i_0_0 * 64 + ax0)
                                    v1 = T.axis.spatial(1024 * p, j_0_0 * 128 + (ax1_0 * 512 + ax1_1 * 128 + ax1_2 * 4 + ax1_3))
                                    T.reads(C_shared[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_shared[v0, v1]


# from tvm.script import tir as T
@tvm.script.ir_module
class StaticModule:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float16"], B: T.Buffer[(1024, 1024), "float16"], C: T.Buffer[(1024, 1024), "float16"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "matmulStatic", "tir.noalias": True})
        s0 = T.var("int32")
        s0_1 = T.var("int32")
        s0_2 = T.var("int32")
        s1 = T.var("int32")
        s1_1 = T.var("int32")
        s1_2 = T.var("int32")
        # body
        # with T.block("root")
        C_reindex_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
        C_reindex_shared_wmma_accumulator = T.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.accumulator")
        A_reindex_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
        B_reindex_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
        A_reindex_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_a")
        B_reindex_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_b")
        for ax0_0_0 in T.thread_binding(16, thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step":16, "pragma_unroll_explicit":1}):
            for ax1_0_0 in T.thread_binding(8, thread="blockIdx.x"):
                for ax0_0_1_ax1_0_1_fused in T.thread_binding(4, thread="threadIdx.y"):
                    for ax0_0_2_init, ax1_0_2_init, ax0_0_3_init, ax1_0_3_init in T.grid(1, 2, 2, 2):
                        with T.block("C_o_init"):
                            v0_o = T.axis.spatial(64, ax0_0_0 * 4 + ax0_0_1_ax1_0_1_fused // 2 * 2 + ax0_0_2_init * 2 + ax0_0_3_init)
                            v1_o = T.axis.spatial(64, ax1_0_0 * 8 + ax0_0_1_ax1_0_1_fused % 2 * 4 + ax1_0_2_init * 2 + ax1_0_3_init)
                            T.reads()
                            T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "warp_execution":1})
                            C_1 = T.match_buffer(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.accumulator", offset_factor=16)
                            T.evaluate(T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // 256 + C_1.elem_offset % 256 // 16, T.float32(0), dtype="handle"))
                    for ax2_0_0 in T.serial(16):
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(8):
                                        with T.block("A_reindex_shared"):
                                            v0 = T.axis.spatial(1024, ax0_0_0 * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) // 64)
                                            v1 = T.axis.spatial(1024, ax2_0_0 * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 8 + ax0_ax1_fused_3) % 64)
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                            A_reindex_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(16):
                            for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(4):
                                        with T.block("B_reindex_shared"):
                                            v0 = T.axis.spatial(1024, ax2_0_0 * 64 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 128)
                                            v1 = T.axis.spatial(1024, ax1_0_0 * 128 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 128)
                                            T.reads(B[v0, v1])
                                            T.writes(B_reindex_shared[v0, v1])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                            B_reindex_shared[v0, v1] = B[v0, v1]
                        for ax2_0_1 in T.serial(4):
                            for ax0_0, ax1_0 in T.grid(2, 1):
                                with T.block("A_reindex_shared_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(64, ax0_0_0 * 4 + ax0_0_1_ax1_0_1_fused // 2 * 2 + ax0_0)
                                    v1_o = T.axis.spatial(64, ax2_0_0 * 4 + ax2_0_1)
                                    T.reads(A_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    A_1 = T.match_buffer(A_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[s1, s0], scope="shared", offset_factor=16)
                                    C_2 = T.match_buffer(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                    T.evaluate(T.tvm_load_matrix_sync(C_2.data, 16, 16, 16, C_2.elem_offset // 256 + C_2.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_1.data, A_1.elem_offset, s1 * 16, 1, dtype="handle"), s1, "row_major", dtype="handle"))
                            for ax0_0, ax1_0 in T.grid(1, 4):
                                with T.block("B_reindex_shared_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(64, ax2_0_0 * 4 + ax2_0_1)
                                    v1_o = T.axis.spatial(64, ax1_0_0 * 8 + ax0_0_1_ax1_0_1_fused % 2 * 4 + ax1_0)
                                    T.reads(B_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    A_2 = T.match_buffer(B_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[s1_1, s0_1], scope="shared", offset_factor=16)
                                    C_3 = T.match_buffer(B_reindex_shared_wmma_matrix_b[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                    T.evaluate(T.tvm_load_matrix_sync(C_3.data, 16, 16, 16, C_3.elem_offset // 256 + C_3.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_2.data, A_2.elem_offset, s1_1 * 16, 1, dtype="handle"), s1_1, "row_major", dtype="handle"))
                            for ax0_0_2, ax1_0_2, ax2_0_2, ax0_0_3, ax1_0_3 in T.grid(1, 2, 1, 2, 2):
                                with T.block("C_o_update"):
                                    v0_o = T.axis.spatial(64, ax0_0_0 * 4 + ax0_0_1_ax1_0_1_fused // 2 * 2 + ax0_0_2 * 2 + ax0_0_3)
                                    v1_o = T.axis.spatial(64, ax1_0_0 * 8 + ax0_0_1_ax1_0_1_fused % 2 * 4 + ax1_0_2 * 2 + ax1_0_3)
                                    v2_o = T.axis.reduce(64, ax2_0_0 * 4 + ax2_0_1 + ax2_0_2)
                                    T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], B_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.writes(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "warp_execution":1})
                                    A_3 = T.match_buffer(A_reindex_shared_wmma_matrix_a[v0_o * 16 : v0_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_reindex_shared_wmma_matrix_b[v2_o * 16 : v2_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                    C_4 = T.match_buffer(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.accumulator", offset_factor=16)
                                    T.evaluate(T.tvm_mma_sync(C_4.data, C_4.elem_offset // 256 + C_4.elem_offset % 256 // 16, A_3.data, A_3.elem_offset // 256 + A_3.elem_offset % 256 // 16, B_1.data, B_1.elem_offset // 256 + B_1.elem_offset % 256 // 16, C_4.data, C_4.elem_offset // 256 + C_4.elem_offset % 256 // 16, dtype="handle"))
                    for ax0_0, ax1_0 in T.grid(2, 4):
                        with T.block("C_reindex_shared_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(64, ax0_0_0 * 4 + ax0_0_1_ax1_0_1_fused // 2 * 2 + ax0_0)
                            v1_o = T.axis.spatial(64, ax1_0_0 * 8 + ax0_0_1_ax1_0_1_fused % 2 * 4 + ax1_0)
                            T.reads(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            T.writes(C_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                            A_4 = T.match_buffer(C_reindex_shared_wmma_accumulator[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", scope="wmma.accumulator", offset_factor=16)
                            C_5 = T.match_buffer(C_reindex_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16], [16, 16], dtype="float16", strides=[s1_2, s0_2], scope="shared", offset_factor=16)
                            T.evaluate(T.tvm_store_matrix_sync(A_4.data, 16, 16, 16, A_4.elem_offset // 256 + A_4.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_5.data, C_5.elem_offset, s1_2 * 16, 2, dtype="handle"), s1_2, "row_major", dtype="handle"))
                for ax0, ax1_0 in T.grid(64, 1):
                    for ax1_1 in T.thread_binding(4, thread="threadIdx.y"):
                        for ax1_2 in T.thread_binding(32, thread="threadIdx.x"):
                            for ax1_3 in T.vectorized(4):
                                with T.block("C_reindex_shared"):
                                    T.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 4 + ax1_3 < 128)
                                    v0 = T.axis.spatial(1024, ax0_0_0 * 64 + ax0)
                                    v1 = T.axis.spatial(1024, ax1_0_0 * 128 + (ax1_0 * 512 + ax1_1 * 128 + ax1_2 * 4 + ax1_3))
                                    T.reads(C_reindex_shared[v0, v1])
                                    T.writes(C[v0, v1])
                                    C[v0, v1] = C_reindex_shared[v0, v1]


def test_dyn_module():
    matmul_mod = tvm.build(DynModule, target="cuda")
    dev = tvm.cuda(0)
    A_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    B_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float16"), dev)
    device = torch.device('cuda:0')
    C_np = torch.tensor(A_np).to(device) @ torch.tensor(B_np).to(device)
    matmul_mod(A_nd, B_nd, C_nd, 1, 1, 1)
    np.testing.assert_allclose(C_np.detach().cpu().numpy(), C_nd.numpy(), atol=2.0)
    num_flop = 2 * 1024 * 1024 * 1024
    evaluator = matmul_mod.time_evaluator("matmul", dev, number=10)
    print("dynamic matmul running time: %f GFLOPS\n" % (num_flop / evaluator(A_nd, B_nd, C_nd, 1, 1, 1).mean / 1e9))


def test_static_module():
    matmul_mod = tvm.build(StaticModule, target="cuda")
    dev = tvm.cuda(0)
    A_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    B_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float16"), dev)
    device = torch.device('cuda:0')
    C_np = torch.tensor(A_np).to(device) @ torch.tensor(B_np).to(device)
    matmul_mod(A_nd, B_nd, C_nd)
    np.testing.assert_allclose(C_np.detach().cpu().numpy(), C_nd.numpy(), atol=2.0)
    num_flop = 2 * 1024 * 1024 * 1024
    evaluator = matmul_mod.time_evaluator("matmulStatic", dev, number=10)
    print("static matmul running time: %f GFLOPS\n" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))


if __name__ == "__main__":
    test_dyn_module()
    test_static_module()
