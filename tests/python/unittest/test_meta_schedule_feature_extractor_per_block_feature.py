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
            "alloc_inner_prod_local",
            "alloc_inner_prod_shared",
            "alloc_inner_prod_global",
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
        print(sch.mod.script())
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
    # Group 1.2: vectorize, unroll, parallel, GPU
    assert_allclose(
        actual=f[16:57],
        desired=[
            # fmt: off
            # vectorize
            1.0, 3.169924, 3.169924, 0, 0, 0, 0, 0, 0, 0, 1,
            # unroll
            1.0, 9.002815, 9.002815, 0, 0, 0, 0, 0, 0, 0, 1,
            # parallel
            1.58496, 11.0007, 6.022368, 0, 0, 0, 0, 0, 0, 0, 1,
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            0.0, 1, 1, 1, 1, 1, 1, 1,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Buffer A
    assert_allclose(
        actual=f[57:75],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            29, 20, 27, 14,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            4.087463, 7.0552826, 3.169925,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            26, 17, 24, 11.0007038,
            # stride
            9.002815,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Buffer C
    assert_allclose(
        actual=f[75:93],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            0, 0, 1,
            # bytes, unique_bytes, lines, unique_lines
            29, 20.000001907348633, 27, 14.00008773803711,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            0, 1, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            1.6147098441152081, 3.2094533656289497, 1,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            29.00000000268723, 20.000001375860553, 27.000000010748916, 14.000088052430122,
            # stride
            9.002815246582031,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Buffer B
    assert_allclose(
        actual=f[93:111],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            29, 20.000001907348633, 19.000001907348633, 14.00008773803711,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            1.0, 3.700439691543579, 4.087462902069092,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            25.0, 16.000022888183594, 15.000043869018555, 10.001408194392809,
            # stride
            0.0,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.4 - 2.5: Dummy padding
    assert_allclose(
        actual=f[111:147],
        desired=[0.0] * (18 * 2),
        rtol=1e-5,
        atol=1e-5,
    )


def test_cpu_fusion():
    # pylint: disable=all
    @T.prim_func
    def func(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [64, 32], dtype="float32")
        B = T.match_buffer(b, [64, 32], dtype="float32")
        C = T.match_buffer(c, [64, 32], dtype="float32")
        for i, j in T.grid(64, 32):  # type: ignore
            with T.block():
                T.reads([A[i, j], B[i, j]])  # type: ignore
                T.writes([B[i, j], C[i, j]])  # type: ignore
                with T.block("B"):
                    T.reads([A[i, j]])  # type: ignore
                    T.writes([B[i, j]])  # type: ignore
                    B[i, j] = A[i, j]  # type: ignore
                with T.block("C"):
                    T.reads([B[i, j]])  # type: ignore
                    T.writes([C[i, j]])  # type: ignore
                    C[i, j] = B[i, j]  # type: ignore

    # pylint: enable=all

    def _create_schedule():
        return tir.Schedule(func, debug_mask="all")

    extractor = ms.feature_extractor.PerBlockFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("llvm")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    print(feature.shape)
    assert feature.shape == (2, N_FEATURES)


if __name__ == "__main__":
    # test_cpu_matmul()
    test_cpu_fusion()
