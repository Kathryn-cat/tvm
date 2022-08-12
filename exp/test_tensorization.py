import argparse
import os

import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm import relay
from tvm.meta_schedule.testing import relay_workload
from tvm.script import tir as T

# python/tvm/tir/tensor_intrin/cuda.py
from tvm.tir.tensor_intrin import cuda as _


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--use_tensorization", action="store_true")
    parser.add_argument("--rpc_host", type=str)
    return parser.parse_args()


ARGS = _parse_args()


@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (4096, 4096), "float16")
        B = T.match_buffer(b, (4096, 4096), "float16")
        C = T.match_buffer(c, (4096, 4096), "float16")
        for i, j, k in T.grid(4096, 4096, 4096):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0.0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def get_search_space():
    context = ms.TuneContext(
        mod=Matmul,
        target=tvm.target.Target("nvidia/nvidia-v100"),
        space_generator=ms.space_generator.PostOrderApply(),
        sch_rules=ms.default_config._DefaultCUDATensorCore.schedule_rules(),
        postprocs=ms.default_config._DefaultCUDATensorCore.postprocs(),
    )
    design_spaces = context.generate_design_space()
    for i, sch in enumerate(design_spaces):
        print(f"design space: {i}")
        print(sch.mod.script())
        print(sch.trace)
        print()


def test_cuda_matmul():
    target = tvm.target.Target("nvidia/nvidia-v100")
    runner = ms.runner.RPCRunner(
        rpc_config=ms.runner.RPCConfig(
            tracker_host=ARGS.rpc_host,
            tracker_port=4445,
            tracker_key="p3.2xlarge",
            session_timeout_sec=100,
        ),
        max_workers=os.cpu_count(),
    )
    with ms.Profiler() as profiler:
        sch: tvm.tir.Schedule = ms.tune_tir(
            mod=Matmul,
            target=target,
            config=ms.TuneConfig(
                num_trials_per_iter=32,
                max_trials_per_task=1000,
                max_trials_global=1000,
            ),
            runner=runner,
            sch_rules=ms.default_config._DefaultCUDATensorCore.schedule_rules,
            postprocs=ms.default_config._DefaultCUDATensorCore.postprocs,
            work_dir=ARGS.work_dir,
            use_tensorization=ARGS.use_tensorization,
        )
    print(profiler.table())
    print(sch.trace)
    print(sch.mod.script())


def test_cuda_tensor_core(model_name, input_shape):
    """Integration tests of auto tensorization with CUDA tensor core"""
    target = tvm.target.Target("nvidia/nvidia-v100")
    dev = tvm.cuda()
    if model_name.startswith("bert"):
        data = tvm.nd.array(np.random.randint(0, 30521, size=input_shape), dev)  # embedding size
    else:
        data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), dev)

    mod, params, (input_name, _, _) = relay_workload.get_network(
        model_name,
        input_shape,
        cache_dir="/home/ubuntu/models/",
    )
    seq = tvm.transform.Sequential(
        [
            relay.transform.ToMixedPrecision(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    def convert_layout(mod):
        seq = tvm.transform.Sequential(
            [relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"]})]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        return mod

    with ms.Profiler() as profiler:
        rt_mod1: tvm.runtime.Module = ms.tune_relay(
            mod=convert_layout(mod),
            params=params,
            target=target,
            config=ms.TuneConfig(
                num_trials_per_iter=32,
                max_trials_per_task=200,
                max_trials_global=0,
            ),
            sch_rules=ms.default_config._DefaultCUDATensorCore.schedule_rules,
            postprocs=ms.default_config._DefaultCUDATensorCore.postprocs,
            work_dir="/home/ubuntu/logs/log_database_1",
        )

    # Compile without meta-scheduler for correctness check
    with tvm.transform.PassContext(opt_level=0):
        rt_mod2 = relay.build(mod, target=target, params=params)

    def get_output(data, lib):
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, data)
        module.run()
        return module.get_output(0).numpy()

    # Check correctness
    actual_output = get_output(data, rt_mod1)
    expected_output = get_output(data, rt_mod2)
    assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=2e-2)


if __name__ == "__main__":
    # test_cuda_tensor_core("bert_base", (8, 128))
    # test_cuda_matmul()
    get_search_space()
