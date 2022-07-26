"""General Matrix-to-matrix Multiply tensor intrinsics in TensorIR"""

import logging
import sys
import tempfile

import tvm.topi.testing
from tvm.meta_schedule import tune_tir
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.meta_schedule.tune import TuneConfig
from tvm.script import tir as T
from tvm.target.target import Target
from tvm.tir import Schedule, tensor_intrin

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods,invalid-name
# pylint: disable=no-self-argument,no-member,protected-access


@tvm.script.ir_module
class GEMM:
    @T.prim_func
    def main(
        A: T.Buffer[(1024, 1024), "float16"],
        B: T.Buffer[(1024, 1024), "float16"],
        C: T.Buffer[(1024, 1024), "float16"],
    ):
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        for ax0, ax1, ax2 in T.grid(1024, 1024, 1024):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [ax0, ax1, ax2])
                with T.init():
                    C[i, j] = T.float16(0)
                C[i, j] += A[i, k] * B[k, j]


mod = GEMM
target = Target("nvidia/nvidia-v100")
config = TuneConfig(
    num_trials_per_iter=64,
    max_trials_per_task=1000,
    max_trials_global=2000,
    search_strategy_config={
        "population_size": 2048,
        "init_measured_ratio": 0.2,
        "init_min_unmeasured": 50,
        "genetic_num_iters": 3,
        "genetic_mutate_prob": 0.85,
        "genetic_max_fail_count": 10,
        "eps_greedy": 0.05,
    },
)


class DefaultTensorCore:
    @staticmethod
    def _sch_rules():
        from tvm.meta_schedule import (
            schedule_rule as M,  # pylint: disable=import-outside-toplevel
        )

        return [
            M.MultiLevelTilingTensorCore(
                intrin_groups=[
                    tensor_intrin.get_wmma_intrin_group("shared", "float16", "float16", False)
                ],
                structure="SSSRRSRS",
                tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y"],
                max_innermost_factor=4,
                vector_load_lens=[1, 2, 3, 4],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=M.ReuseType(
                    req="must",
                    levels=[2],
                    scope="shared",
                ),
            ),
            M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 4, 8],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=M.ReuseType(
                    req="must",
                    levels=[3],
                    scope="local",
                ),
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]

    @staticmethod
    def _postproc():
        from tvm.meta_schedule import (
            postproc as M,  # pylint: disable=import-outside-toplevel
        )

        return [
            M.RewriteCooperativeFetch(),
            M.RewriteUnboundBlock(),
            M.RewriteParallelVectorizeUnroll(),
            M.RewriteReductionBlock(),
            M.RewriteTensorize(),
            M.VerifyGPUCode(),
        ]

    @staticmethod
    def _mutator_probs():
        from tvm.meta_schedule import (
            mutator as M,  # pylint: disable=import-outside-toplevel
        )

        return {
            M.MutateTileSize(): 0.9,
            M.MutateUnroll(): 0.1,
        }


with tempfile.TemporaryDirectory() as work_dir:
    sch: Schedule = tune_tir(
        mod=mod,
        target=target,
        config=config,
        work_dir=work_dir,
        space=PostOrderApply(),
        sch_rules=DefaultTensorCore._sch_rules,
        postprocs=DefaultTensorCore._postproc,
        mutator_probs=DefaultTensorCore._mutator_probs,
        num_threads=None,
    )
    if sch is None:
        print("No valid schedule found!")
        sys.exit()

    print(sch.mod.script())
    print("\n".join(sch.trace.as_python()))
