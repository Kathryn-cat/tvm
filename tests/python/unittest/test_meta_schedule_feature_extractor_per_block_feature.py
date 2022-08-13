# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import sys
from typing import Callable, List

import tvm
from numpy.testing import assert_allclose
from tvm import meta_schedule as ms
from tvm import te, tir
from tvm.script import tir as T

N_FEATURES = 172


@T.prim_func
def matmul(
    A: T.Buffer[(512, 512), "float32"],
    B: T.Buffer[(512, 512), "float32"],
    C: T.Buffer[(512, 512), "float32"],
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("C"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C[i, j], A[i, k], B[k, j])
            T.writes(C[i, j])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


def _make_context(target) -> ms.TuneContext:
    return ms.TuneContext(
        target=target,
        num_threads=1,
    )


def _make_candidate(f_sch: Callable[[], tir.Schedule]) -> ms.MeasureCandidate:
    return ms.MeasureCandidate(sch=f_sch(), args_info=[])


def _feature_names(  # pylint: disable=invalid-name
    buffers_per_block: int = 5,
    arith_intensity_curve_num_samples: int = 10,
) -> List[str]:
    result = [
        "float_mad",
        "float_addsub",
        "float_mul",
        "float_divmod",
        "float_cmp",
        "float_mathfunc",
        "float_otherfunc",
        "int_mad",
        "int_addsub",
        "int_mul",
        "int_divmod",
        "int_cmp",
        "int_mathfunc",
        "int_otherfunc",
        "bool_op",
        "select_op",
        "vec_num",
        "vec_prod",
        "vec_len",
        "vec_type.kPosNone",
        "vec_type.kPosInnerSpatial",
        "vec_type.kPosMiddleSpatial",
        "vec_type.kPosOuterSpatial",
        "vec_type.kPosInnerReduce",
        "vec_type.kPosMiddleReduce",
        "vec_type.kPosOuterReduce",
        "vec_type.kPosMixed",
        "unroll_num",
        "unroll_prod",
        "unroll_len",
        "unroll_type.kPosNone",
        "unroll_type.kPosInnerSpatial",
        "unroll_type.kPosMiddleSpatial",
        "unroll_type.kPosOuterSpatial",
        "unroll_type.kPosInnerReduce",
        "unroll_type.kPosMiddleReduce",
        "unroll_type.kPosOuterReduce",
        "unroll_type.kPosMixed",
        "parallel_num",
        "parallel_prod",
        "parallel_len",
        "parallel_type.kPosNone",
        "parallel_type.kPosInnerSpatial",
        "parallel_type.kPosMiddleSpatial",
        "parallel_type.kPosOuterSpatial",
        "parallel_type.kPosInnerReduce",
        "parallel_type.kPosMiddleReduce",
        "parallel_type.kPosOuterReduce",
        "parallel_type.kPosMixed",
        "is_gpu",
        "blockIdx_x_len",
        "blockIdx_y_len",
        "blockIdx_z_len",
        "threadIdx_x_len",
        "threadIdx_y_len",
        "threadIdx_z_len",
        "vthread_len",
    ]
    for i in range(buffers_per_block):
        result.extend(
            f"B{i}.{s}"
            for s in [
                "acc_type.kRead",
                "acc_type.kWrite",
                "acc_type.kReadWrite",
                "bytes",
                "unique_bytes",
                "lines",
                "unique_lines",
                "reuse_type.kLoopMultipleRead",
                "reuse_type.kSerialMultipleReadWrite",
                "reuse_type.kNoReuse",
                "reuse_dis_iter",
                "reuse_dis_bytes",
                "reuse_ct",
                "bytes_d_reuse_ct",
                "unique_bytes_d_reuse_ct",
                "lines_d_reuse_ct",
                "unique_lines_d_reuse_ct",
                "stride",
            ]
        )
    result.extend(f"arith_intensity_curve_{i}" for i in range(arith_intensity_curve_num_samples))
    result.extend(
        [
            "alloc_size_local",
            "alloc_size_shared",
            "alloc_size_global",
            "alloc_prod_local",
            "alloc_prod_shared",
            "alloc_prod_global",
            "alloc_outer_prod_local",
            "alloc_outer_prod_shared",
            "alloc_outer_prod_global",
            "alloc_inner_prod",
            "outer_prod",
            "num_loops",
            "auto_unroll_max_step",
        ]
    )
    # 57 + 18 * 5 + 10 + 12 + 3
    # assert len(result) == N_FEATURES
    return result


def _zip_feature(feature, names):
    assert feature.ndim == 1
    # assert feature.shape[0] == N_FEATURES
    # assert len(names) == N_FEATURES
    return list(zip(names, feature))


def _print_feature(feature, st, ed):  # pylint: disable=invalid-name
    named_feature = _zip_feature(feature, _feature_names())
    for k, v in named_feature[st:ed]:
        print("\t", k, v)


def test_cpu_matmul():
    def _create_schedule():
        func = matmul
        sch = tir.Schedule(func, debug_mask="all")
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        i_o, i_i = sch.split(i, factors=[None, 16])  # outer: 32
        j_o, j_i = sch.split(j, factors=[None, 8])  # outer: 64
        sch.reorder(i_o, j_o, k, j_i, i_i)
        sch.vectorize(j_i)
        sch.parallel(i_o)
        sch.parallel(j_o)
        sch.unroll(k)
        return sch

    extractor = ms.feature_extractor.PerBlockFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    assert feature.shape == (1, N_FEATURES)
    f = feature[0]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:16],
        # fmt: off
        desired=[
            # float math ops
            0, 27, 27, 0, 0, 0, 0,
            # int math ops
            0, 29, 29, 0, 0, 0, 0,
            # bool/select ops
            0, 0,
        ],
        # fmt: on
        rtol=1e-5,
        atol=1e-5,
    )
    _print_feature(f, 0, 16)


if __name__ == "__main__":
    test_cpu_matmul()
