from typing import List
import argparse

import numpy as np
import torch

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _
from tvm.tir.tensor_intrin.cuda import get_wmma_store_intrin
from tvm.meta_schedule.postproc import Postproc


# fixed shape matmul
@T.prim_func
def matmulStatic(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "matmulStatic", "tir.noalias": True})
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
    A = T.match_buffer(a, (1024 * m, 1024 * n), "float16")
    B = T.match_buffer(b, (1024 * n, 1024 * p), "float16")
    C = T.match_buffer(c, (1024 * m, 1024 * p), "float16")
    for i, j, k in T.grid(1024 * m, 1024 * n, 1024 * p):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float16(0.0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def schedule_matmul(sch: tir.Schedule) -> None:
    b_C = sch.get_block("C")
    i, j, k = sch.get_loops(block=b_C)
    i0, i1 = sch.split(loop=i, factors=[None, 16])
    j0, j1 = sch.split(loop=j, factors=[None, 16])
    k0, k1 = sch.split(loop=k, factors=[None, 16])
    sch.reorder(i0, j0, k0, i1, j1, k1)
    b_mm = sch.blockize(i1)
    # loop max size is m, n, p
    l_g1, l_g2, l_g3 = sch.get_loops(b_mm)
    cand, prob = categ(5)
    # split l_g1
    v1_g1 = sch.sample_categorical(candidates=cand, probs=prob)
    v1_g1 = 4
    l_g11, l_g1r = sch.split(loop=l_g1, factors=[None, v1_g1])
    res = sch.sample_perfect_tile(loop=l_g1r, n=3, max_innermost_factor=4)
    res = [2, 1, 2]
    l_g12, l_g13, l_g14 = sch.split(loop=l_g1r, factors=[*res])
    # split l_g2
    v1_g2 = sch.sample_categorical(candidates=cand, probs=prob)
    v1_g2 = 8
    l_g21, l_g2r = sch.split(loop=l_g2, factors=[None, v1_g2])
    res = sch.sample_perfect_tile(loop=l_g2r, n=3, max_innermost_factor=4)
    res = [2, 2, 2]
    l_g22, l_g23, l_g24 = sch.split(loop=l_g2r, factors=[*res])
    # split l_g3
    v1_g3 = sch.sample_categorical(candidates=cand, probs=prob)
    v1_g3 = 4
    l_g31, l_g3r = sch.split(loop=l_g3, factors=[None, v1_g3])
    res = sch.sample_perfect_tile(loop=l_g3r, n=2, max_innermost_factor=4)
    res = [4, 1]
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
    v_s = 4
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
    v_s = 8
    l_s1, l_s2, l_s3, l_s4 = sch.split(loop=l_s, factors=[None, sch.get(l_g3).extent, 32, v_s])
    sch.vectorize(loop=l_s4)
    sch.bind(loop=l_s3, thread_axis="threadIdx.x")
    sch.bind(loop=l_s2, thread_axis="threadIdx.y")
    # cooperative fetch for shared memory for B
    _, _, _, _, l_s = sch.get_loops(block=B_shared)
    v_s = sch.sample_categorical(candidates=cand, probs=prob)
    v_s = 4
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
    b_ti = sch.get_block("C_o_init")
    sch.tensorize(block_or_loop=b_ti, tensor_intrin="wmma_fill_16x16x16_f16")
    b_tc = sch.get_block("C_o_update")
    sch.tensorize(block_or_loop=b_tc, tensor_intrin="wmma_sync_16x16x16_f16f16f16")


def categ(k):
    """should tweak weights in the future to account for independence"""
    arr = np.arange(k + 1)
    cand = 2**arr
    cand = [int(i) for i in cand]
    prob = np.ones(k + 1) / (k + 1)
    return cand, list(prob)


def test():
        sch = tvm.tir.Schedule(matmulStatic)
        schedule_matmul(sch)
        # sch.mod.show()
        matmul_mod = tvm.build(sch.mod, target="cuda")

        # evaluate the running time
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
        # print("calculation is correct")
        num_flop = 2 * 1024 * 1024 * 1024
        evaluator = matmul_mod.time_evaluator("matmulStatic", dev, number=10)
        print("matmul running time: %f GFLOPS\n" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))


if __name__ == "__main__":
    test()
