import numpy as np
import torch

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _


@tvm.script.ir_module
class StaticModule:
    @T.prim_func
    def func(A: T.Buffer[1048576, "float16"], B: T.Buffer[1048576, "float16"], C: T.Buffer[1048576, "float16"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "matmulStatic", "tir.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        T.preflattened_buffer(A, [1024, 1024], dtype="float16", data=A.data)
        T.preflattened_buffer(B, [1024, 1024], dtype="float16", data=B.data)
        T.preflattened_buffer(C, [1024, 1024], dtype="float16", data=C.data)
        # body
        T.attr(blockIdx_y, "pragma_auto_unroll_max_step", 16)
        T.attr(blockIdx_y, "pragma_unroll_explicit", 1)
        T.launch_thread(blockIdx_y, 16)
        T.launch_thread(blockIdx_x, 8)
        T.launch_thread(threadIdx_y, 4)
        T.launch_thread(threadIdx_x, 32)
        C_shared = T.allocate([8192], "float16", "shared")
        C_shared_1 = T.buffer_decl([8192], dtype="float16", data=C_shared, scope="shared")
        with T.allocate([2048], "float16", "wmma.accumulator") as C_shared_wmma_accumulator:
            for j_0_2_init, i_0_3_init, j_0_3_init in T.grid(2, 2, 2):
                T.evaluate(T.tvm_fill_fragment(C_shared_wmma_accumulator, 16, 16, 16, (0 * 2048 + i_0_3_init * 1024 + j_0_2_init * 32 + j_0_3_init * 16) // 256 + (0 * 2048 + i_0_3_init * 1024 + j_0_2_init * 32 + j_0_3_init * 16) % 256 // 16, T.float32(0), dtype="handle"))
            for k_0_0 in T.serial(16):
                A_shared = T.allocate([4608], "float16", "shared")
                A_shared_1 = T.buffer_decl([4608], dtype="float16", data=A_shared, scope="shared")
                A_shared_2 = T.decl_buffer([64, 64], dtype="float16", data=A_shared, strides=[72, 1], scope="shared")
                B_shared = T.allocate([8704], "float16", "shared")
                B_shared_1 = T.buffer_decl([8704], dtype="float16", data=B_shared, scope="shared")
                B_shared_2 = T.decl_buffer([64, 128], dtype="float16", data=B_shared, strides=[136, 1], scope="shared")
                for ax0_ax1_fused_0 in T.serial(4):
                    for ax0_ax1_fused_3 in T.vectorized(8):
                        A_shared_1[ax0_ax1_fused_0 * 1152 + threadIdx_y * 288 + (threadIdx_x * 8 + ax0_ax1_fused_3) // 64 * 72 + (threadIdx_x * 8 + ax0_ax1_fused_3) % 64] = A[blockIdx_y * 65536 + ax0_ax1_fused_0 * 16384 + threadIdx_y * 4096 + (threadIdx_x * 8 + ax0_ax1_fused_3) // 64 * 1024 + k_0_0 * 64 + (threadIdx_x * 8 + ax0_ax1_fused_3) % 64]
                for ax0_ax1_fused_0 in T.serial(16):
                    for ax0_ax1_fused_3 in T.vectorized(4):
                        B_shared_1[ax0_ax1_fused_0 * 544 + (threadIdx_x * 4 + ax0_ax1_fused_3) // 128 * 136 + threadIdx_y * 136 + (threadIdx_x * 4 + ax0_ax1_fused_3) % 128] = B[k_0_0 * 65536 + ax0_ax1_fused_0 * 4096 + (threadIdx_x * 4 + ax0_ax1_fused_3) // 128 * 1024 + threadIdx_y * 1024 + blockIdx_x * 128 + (threadIdx_x * 4 + ax0_ax1_fused_3) % 128]
                for k_0_1 in T.serial(4):
                    B_shared_wmma_matrix_b = T.allocate([1024], "float16", "wmma.matrix_b")
                    A_shared_wmma_matrix_a = T.allocate([512], "float16", "wmma.matrix_a")
                    for ax0_0 in T.serial(2):
                        T.evaluate(T.tvm_load_matrix_sync(A_shared_wmma_matrix_a, 16, 16, 16, ax0_0 * 256 // 256 + ax0_0 * 256 % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_shared, threadIdx_y // 2 * 2304 + ax0_0 * 1152 + k_0_1 * 16, 72 * 16, 1, dtype="handle"), 72, "row_major", dtype="handle"))
                    for ax1_0 in T.serial(4):
                        T.evaluate(T.tvm_load_matrix_sync(B_shared_wmma_matrix_b, 16, 16, 16, ax1_0 * 16 // 256 + ax1_0 * 16 % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), B_shared, k_0_1 * 2176 + threadIdx_y % 2 * 64 + ax1_0 * 16, 136 * 16, 1, dtype="handle"), 136, "row_major", dtype="handle"))
                    for j_0_2, i_0_3, j_0_3 in T.grid(2, 2, 2):
                        T.evaluate(T.tvm_mma_sync(C_shared_wmma_accumulator, (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) // 256 + (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) % 256 // 16, A_shared_wmma_matrix_a, (0 * 512 + i_0_3 * 256 + 0 * 16) // 256 + (0 * 512 + i_0_3 * 256 + 0 * 16) % 256 // 16, B_shared_wmma_matrix_b, (0 * 1024 + j_0_2 * 32 + j_0_3 * 16) // 256 + (0 * 1024 + j_0_2 * 32 + j_0_3 * 16) % 256 // 16, C_shared_wmma_accumulator, (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) // 256 + (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) % 256 // 16, dtype="handle"))
            for ax0_0, ax1_0 in T.grid(2, 4):
                T.evaluate(T.tvm_store_matrix_sync(C_shared_wmma_accumulator, 16, 16, 16, (ax0_0 * 1024 + ax1_0 * 16) // 256 + (ax0_0 * 1024 + ax1_0 * 16) % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_shared, threadIdx_y // 2 * 4096 + ax0_0 * 2048 + threadIdx_y % 2 * 64 + ax1_0 * 16, 128 * 16, 2, dtype="handle"), 128, "row_major", dtype="handle"))
        for ax0 in T.serial(64):
            for ax1_3 in T.vectorized(4):
                if ((0 * 4 + threadIdx_y) * 32 + threadIdx_x) * 4 + ax1_3 < 128:
                    C[blockIdx_y * 65536 + ax0 * 1024 + blockIdx_x * 128 + threadIdx_y * 128 + threadIdx_x * 4 + ax1_3] = C_shared_1[ax0 * 128 + threadIdx_y * 128 + threadIdx_x * 4 + ax1_3]


@tvm.script.ir_module
class DynModule:
    @T.prim_func
    def func(a: T.handle, b: T.handle, c: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        A = T.match_buffer(a, [1024 * m * (1024 * n)], dtype="float16")
        B = T.match_buffer(b, [1024 * n * (1024 * p)], dtype="float16")
        C = T.match_buffer(c, [1024 * m * (1024 * p)], dtype="float16")
        T.preflattened_buffer(A, [1024 * m, 1024 * n], dtype="float16", data=A.data)
        T.preflattened_buffer(B, [1024 * n, 1024 * p], dtype="float16", data=B.data)
        T.preflattened_buffer(C, [1024 * m, 1024 * p], dtype="float16", data=C.data)
        # body
        T.attr(blockIdx_y, "pragma_auto_unroll_max_step", 16)
        T.attr(blockIdx_y, "pragma_unroll_explicit", 1)
        T.launch_thread(blockIdx_y, m * 16)
        T.launch_thread(blockIdx_x, n * 8)
        T.launch_thread(threadIdx_y, 4)
        T.launch_thread(threadIdx_x, 32)
        C_shared = T.allocate([8192], "float16", "shared")
        C_shared_1 = T.buffer_decl([8192], dtype="float16", data=C_shared, scope="shared")
        with T.allocate([2048], "float16", "wmma.accumulator") as C_shared_wmma_accumulator:
            for j_0_2_init, i_0_3_init, j_0_3_init in T.grid(2, 2, 2):
                T.evaluate(T.tvm_fill_fragment(C_shared_wmma_accumulator, 16, 16, 16, (0 * 2048 + i_0_3_init * 1024 + j_0_2_init * 32 + j_0_3_init * 16) // 256 + (0 * 2048 + i_0_3_init * 1024 + j_0_2_init * 32 + j_0_3_init * 16) % 256 // 16, T.float32(0), dtype="handle"))
            for k_0_0 in T.serial(p * 16):
                A_shared = T.allocate([4608], "float16", "shared")
                A_shared_1 = T.buffer_decl([4608], dtype="float16", data=A_shared, scope="shared")
                A_shared_2 = T.decl_buffer([64, 64], dtype="float16", data=A_shared, strides=[72, 1], scope="shared")
                B_shared = T.allocate([8704], "float16", "shared")
                B_shared_1 = T.buffer_decl([8704], dtype="float16", data=B_shared, scope="shared")
                B_shared_2 = T.decl_buffer([64, 128], dtype="float16", data=B_shared, strides=[136, 1], scope="shared")
                for ax0_ax1_fused_0 in T.serial(4):
                    for ax0_ax1_fused_3 in T.vectorized(8):
                        A_shared_1[ax0_ax1_fused_0 * 1152 + threadIdx_y * 288 + (threadIdx_x * 8 + ax0_ax1_fused_3) // 64 * 72 + (threadIdx_x * 8 + ax0_ax1_fused_3) % 64] = A[k_0_0 * 64 + (blockIdx_y * 64 + ax0_ax1_fused_0 * 16 + threadIdx_y * 4 + (threadIdx_x * 8 + ax0_ax1_fused_3) // 64) * (n * 1024) + (threadIdx_x * 8 + ax0_ax1_fused_3) % 64]
                for ax0_ax1_fused_0 in T.serial(16):
                    for ax0_ax1_fused_3 in T.vectorized(4):
                        B_shared_1[ax0_ax1_fused_0 * 544 + (threadIdx_x * 4 + ax0_ax1_fused_3) // 128 * 136 + threadIdx_y * 136 + (threadIdx_x * 4 + ax0_ax1_fused_3) % 128] = B[blockIdx_x * 128 + (k_0_0 * 64 + ax0_ax1_fused_0 * 4 + (threadIdx_x * 4 + ax0_ax1_fused_3) // 128 + threadIdx_y) * (p * 1024) + (threadIdx_x * 4 + ax0_ax1_fused_3) % 128]
                for k_0_1 in T.serial(4):
                    B_shared_wmma_matrix_b = T.allocate([1024], "float16", "wmma.matrix_b")
                    A_shared_wmma_matrix_a = T.allocate([512], "float16", "wmma.matrix_a")
                    for ax0_0 in T.serial(2):
                        T.evaluate(T.tvm_load_matrix_sync(A_shared_wmma_matrix_a, 16, 16, 16, ax0_0 * 256 // 256 + ax0_0 * 256 % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), A_shared, threadIdx_y // 2 * 2304 + ax0_0 * 1152 + k_0_1 * 16, 72 * 16, 1, dtype="handle"), 72, "row_major", dtype="handle"))
                    for ax1_0 in T.serial(4):
                        T.evaluate(T.tvm_load_matrix_sync(B_shared_wmma_matrix_b, 16, 16, 16, ax1_0 * 16 // 256 + ax1_0 * 16 % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), B_shared, k_0_1 * 2176 + threadIdx_y % 2 * 64 + ax1_0 * 16, 136 * 16, 1, dtype="handle"), 136, "row_major", dtype="handle"))
                    for j_0_2, i_0_3, j_0_3 in T.grid(2, 2, 2):
                        T.evaluate(T.tvm_mma_sync(C_shared_wmma_accumulator, (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) // 256 + (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) % 256 // 16, A_shared_wmma_matrix_a, (0 * 512 + i_0_3 * 256 + 0 * 16) // 256 + (0 * 512 + i_0_3 * 256 + 0 * 16) % 256 // 16, B_shared_wmma_matrix_b, (0 * 1024 + j_0_2 * 32 + j_0_3 * 16) // 256 + (0 * 1024 + j_0_2 * 32 + j_0_3 * 16) % 256 // 16, C_shared_wmma_accumulator, (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) // 256 + (0 * 2048 + i_0_3 * 1024 + j_0_2 * 32 + j_0_3 * 16) % 256 // 16, dtype="handle"))
            for ax0_0, ax1_0 in T.grid(2, 4):
                T.evaluate(T.tvm_store_matrix_sync(C_shared_wmma_accumulator, 16, 16, 16, (ax0_0 * 1024 + ax1_0 * 16) // 256 + (ax0_0 * 1024 + ax1_0 * 16) % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), C_shared, threadIdx_y // 2 * 4096 + ax0_0 * 2048 + threadIdx_y % 2 * 64 + ax1_0 * 16, 128 * 16, 2, dtype="handle"), 128, "row_major", dtype="handle"))
        for ax0 in T.serial(64):
            for ax1_3 in T.vectorized(4):
                if ((0 * 4 + threadIdx_y) * 32 + threadIdx_x) * 4 + ax1_3 < 128:
                    C[blockIdx_x * 128 + threadIdx_y * 128 + threadIdx_x * 4 + (blockIdx_y * 64 + ax0) * (p * 1024) + ax1_3] = C_shared_1[ax0 * 128 + threadIdx_y * 128 + threadIdx_x * 4 + ax1_3]


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
