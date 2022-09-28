import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.tir.tensor_intrin import cuda as _


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


# for now, we only consider 1024, 2048, 4096
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
    v1 = sch.sample_categorical(candidates=[1, 2, 4, 8, 16, 32, 64], probs=[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
    v2 = sch.sample_categorical(candidates=[1, 2, 4, 8, 16, 32, 64], probs=[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
    i0, i1, i2 = sch.split(loop=i, factors=[None, v1, 16])
    j0, j1, j2 = sch.split(loop=j, factors=[None, v2, 16])
    k0, k1 = sch.split(loop=k, factors=[None, 16])
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    b_mm = sch.blockize(k1)
    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
    sch.bind(i1, "threadIdx.y")
    b_shared = sch.blockize(j1)
    C_shared = sch.cache_write(b_shared, 0, "shared")
    sch.reverse_compute_at(C_shared, j0)
    A_shared = sch.cache_read(block=b_shared, read_buffer_index=0, storage_scope="shared")
    B_shared = sch.cache_read(block=b_shared, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=A_shared, loop=i1)
    sch.compute_at(block=B_shared, loop=i1)
    b_local = sch.blockize(k0)
    C_local = sch.cache_write(b_local, 0, "local")
    sch.reverse_compute_at(C_local, j1)
    A_local = sch.cache_read(block=b_local, read_buffer_index=1, storage_scope="local")
    B_local = sch.cache_read(block=b_local, read_buffer_index=2, storage_scope="local")
    sch.compute_at(block=A_local, loop=j1)
    sch.compute_at(block=B_local, loop=j1)
    #sch.tensorize(block_or_loop=b_mm, tensor_intrin="wmma_16x16x16_f16") - TODO: argument?

if __name__ == "__main__":
    sch = tvm.tir.Schedule(matmul)
    schedule_matmul(sch)
    sch.mod.show()
    # matmul_mod = tvm.build(sch.mod, target="cuda")

    """
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
                max_trials_global=1000,
            ),
            sch_rules=ms.default_config._DefaultCUDATensorCore.schedule_rules,
            postprocs=ms.default_config._DefaultCUDATensorCore.postprocs,
            work_dir="logs/test-1",
        )
    """
