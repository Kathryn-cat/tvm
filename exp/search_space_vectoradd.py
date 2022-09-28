import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.schedule_rule import PyScheduleRule
from tvm.meta_schedule.utils import derived_object
from tvm.script import tir as T
from tvm.tir.schedule import BlockRV, Schedule
from tvm.tir.tensor_intrin import cuda as _


@T.prim_func
def vectoradd(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "vectoradd", "tir.noalias": True})
    m = T.var("int32")
    A = T.match_buffer(a, (m, ), "float32")
    B = T.match_buffer(b, (m, ), "float32")
    C = T.match_buffer(c, (m, ), "float32")
    for i in T.grid(m):
        with T.block("update"):
            vi = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]

def schedule_vectoradd(sch: tir.Schedule) -> None:
    # question: should we bind block/thread to var?
    block_u = sch.get_block("update")
    i, = sch.get_loops(block=block_u)
    v = sch.sample_categorical(candidates=[1, 2, 4, 8, 16, 32, 64], probs=[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
    i0, i1 = sch.split(i, [None, v])
    sch.bind(i0, "blockIdx.x")
    sch.bind(i1, "threadIdx.x")

@derived_object
class ScheduleFn(PyScheduleRule):
    def _initialize_with_tune_context(self, context: TuneContext) -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV):
        schedule_vectoradd(sch)
        return [sch]


if __name__ == "__main__":
    print(f'original vectoradd')
    print(vectoradd.script())
    sch = tvm.tir.Schedule(vectoradd)
    schedule_vectoradd(sch)
    print(f'new module')
    print(sch.mod.script())
    vectoradd_mod = tvm.build(sch.mod, target="cuda")

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

    # metaschedule tuning
    target = tvm.target.Target("nvidia/geforce-rtx-3090")
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.tune_tir(
            mod=vectoradd,
            target=target,
            config=ms.TuneConfig(
                num_trials_per_iter=32,
                max_trials_per_task=1000,
                max_trials_global=1000,
            ),
            sch_rules=lambda *args: [ScheduleFn()],
            work_dir="vectoradd_sp1",
        )
