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
    assert len(result) == N_FEATURES
    return result


def _zip_feature(feature, names):
    assert feature.ndim == 1
    assert feature.shape[0] == N_FEATURES
    assert len(names) == N_FEATURES
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
    # Group 1: arith, loop
    assert_allclose(
        actual=f[0:57],
        # fmt: off
        desired=[
            # float math ops
            0, 27, 27, 0, 0, 0, 0,
            # int math ops
            0, 29, 29, 0, 0, 0, 0,
            # bool/select ops
            0, 0,
            # vectorize
            1.0, 3.169924, 3.169924, 0, 0, 0, 0, 0, 0, 0, 1,
            # unroll
            1.0, 9.002815, 9.002815, 0, 0, 0, 0, 0, 0, 0, 1,
            # parallel
            1.58496, 11.0007, 6.022368, 0, 0, 0, 0, 0, 0, 0, 1,
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            0.0, 1, 1, 1, 1, 1, 1, 1,
        ],
        # fmt: on
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
    # TODO(@Kathryn-cat): Test group 3 - 5


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
    assert feature.shape == (2, N_FEATURES)
    ## Features for Block(B)
    f = feature[0]
    # Group 1: arith, loop
    assert_allclose(
        actual=f[0:57],
        # fmt: off
        desired=[0.0] * 16
            # vectorize
            + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # unroll
            + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # parallel
            + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            + [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # fmt: on
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Buffer B
    assert_allclose(
        actual=f[57:75],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            13.000176429748535, 13.000176429748535, 7.011227130889893, 7.011227130889893,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            0, 0, 1,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            0, 0, 0,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            14.00008773803711, 14.00008773803711, 8.005624771118164, 8.005624771118164,
            # stride
            1,
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
            0, 1, 0,
            # bytes, unique_bytes, lines, unique_lines
            13.000176429748535, 13.000176429748535, 7.011227130889893, 7.011227130889893,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            0, 0, 1,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            0, 0, 0,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            14.00008773803711, 14.00008773803711, 8.005624771118164, 8.005624771118164,
            # stride
            1,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3 - 2.5: Dummy padding
    assert_allclose(
        actual=f[93:147],
        desired=[0.0] * (18 * 3),
        rtol=1e-5,
        atol=1e-5,
    )
    # TODO(@Kathryn-cat): Test group 3 - 5
    ## Features for Block(C)
    f = feature[1]
    # Group 1: arith, loop
    assert_allclose(
        actual=f[0:57],
        # fmt: off
        desired=[0.0] * 16
            # vectorize
            + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # unroll
            + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # parallel
            + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            + [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # fmt: on
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Buffer B
    assert_allclose(
        actual=f[57:75],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            13.000176429748535, 13.000176429748535, 7.011227130889893, 7.011227130889893,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            0, 0, 1,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            0, 0, 0,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            14.00008773803711, 14.00008773803711, 8.005624771118164, 8.005624771118164,
            # stride
            1,
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
            0, 1, 0,
            # bytes, unique_bytes, lines, unique_lines
            13.000176429748535, 13.000176429748535, 7.011227130889893, 7.011227130889893,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            0, 0, 1,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            0, 0, 0,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            14.00008773803711, 14.00008773803711, 8.005624771118164, 8.005624771118164,
            # stride
            1,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3 - 2.5: Dummy padding
    assert_allclose(
        actual=f[93:147],
        desired=[0.0] * (18 * 3),
        rtol=1e-5,
        atol=1e-5,
    )
    # TODO(@Kathryn-cat): Test group 3 - 5


def test_gpu():
    def _create_schedule():
        func = matmul
        sch = tir.Schedule(func, debug_mask="all")
        c = sch.get_block("C")
        c_local = sch.cache_write(c, 0, "local")
        i, j, k = sch.get_loops(c)
        # pylint: disable=invalid-name
        i0, i1, i2, i3, i4 = sch.split(i, factors=[None, 1, 16, 32, 1])  # outer: 1
        j0, j1, j2, j3, j4 = sch.split(j, factors=[None, 4, 1, 1, 16])  # outer: 8
        k0, k1, k2 = sch.split(k, factors=[None, 1, 2])  # outer: 256
        # pylint: enable=invalid-name
        # fmt: off
        sch.reorder(
            i0, j0,  # S
            i1, j1,  # S
            i2, j2,  # S
            k0,      # R
            k1,      # R
            i3, j3,  # S
            k2,      # R
            i4, j4,  # S
        )
        # fmt: on
        # thread binding
        i0_j0 = sch.fuse(i0, j0)
        i1_j1 = sch.fuse(i1, j1)
        i2_j2 = sch.fuse(i2, j2)
        sch.bind(i0_j0, "blockIdx.x")
        sch.bind(i1_j1, "vthread.x")
        sch.bind(i2_j2, "threadIdx.x")
        # fusion
        sch.reverse_compute_at(c_local, i2_j2)
        # cache read 'A'
        a_shared = sch.cache_read(c, 1, "shared")
        sch.compute_at(a_shared, k0)
        _, _, _, _, a_i, a_j = sch.get_loops(a_shared)
        a_ij = sch.fuse(a_i, a_j)
        _, a_j = sch.split(a_ij, factors=[None, 16])  # outer: 64
        sch.bind(a_j, "threadIdx.x")
        # cache read 'B'
        b_shared = sch.cache_read(c, 2, "shared")
        sch.compute_at(b_shared, k0)
        _, _, _, _, b_i, b_j = sch.get_loops(b_shared)
        b_ij = sch.fuse(b_i, b_j)
        _, b_j = sch.split(b_ij, factors=[None, 16])  # outer: 8
        sch.bind(b_j, "threadIdx.x")
        # auto unroll
        sch.annotate(i0_j0, "pragma_auto_unroll_max_step", tir.IntImm("int32", 1024))
        sch.annotate(i0_j0, "pragma_unroll_explicit", tir.IntImm("int32", 1))
        return sch

    extractor = ms.feature_extractor.PerBlockFeature()
    (feature,) = extractor.extract_from(
        _make_context(tvm.target.Target("cuda")),
        candidates=[_make_candidate(_create_schedule)],
    )
    feature = feature.numpy()
    assert feature.shape == (4, N_FEATURES)
    ### Check feature[0]: BufferStore(A_shared) <= A[...]
    f = feature[0]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:57],
        desired=[
            # fmt: off
            # float math ops
            0, 0, 0, 0, 0, 0, 0,
            # int math ops
            0, 24.000000085991324, 24.000000085991324, 24.000000085991324, 0, 0, 0,
            # bool/select ops
            0, 0,
            # vectorize
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # unroll
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # parallel
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362,
            # fmt: on
            ######
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
            25.000000042995662, 20.000001375860553, 23.00000017198264, 14.000088052430122,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            18.00000550343433, 20.00562591970089, 2.321928094887362,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            23.00000017198264, 18.00000550343433, 21.000000687930438, 12.0003521774803,
            # stride
            12.0003521774803,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Buffer A.shared
    assert_allclose(
        actual=f[75:93],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            0, 1, 0,
            # bytes, unique_bytes, lines, unique_lines
            25.000000042995662, 12.0003521774803, 23.00000017198264, 9.002815015607053,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            6.022367813028454, 11.98049663618346, 8.005624549193879,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            17.000011006847668, 4.087462841250339, 15.000044026886828, 1.584962500721156,
            # stride
            4.087462841250339,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3 - 2.5: Dummy padding
    assert_allclose(
        actual=f[93:147],
        desired=[0] * (18 * 3),
        rtol=1e-5,
        atol=1e-5,
    )
    # TODO(@Kathryn-cat): Test group 3 - 5
    ### Check feature[1]: BufferStore(B_shared) <= B[...]
    f = feature[1]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:57],
        desired=[
            # fmt: off
            # float math ops
            0, 0, 0, 0, 0, 0, 0,
            # int math ops
            0, 21.584962959341485, 21.584962959341485, 21.000000687930438, 0, 0, 0,
            # bool/select ops
            0, 0,
            # vectorize
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # unroll
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # parallel
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362,
            # fmt: on
            ######
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Buffer B
    assert_allclose(
        actual=f[57:75],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            22.00000034396526, 20.000001375860553, 20.000001375860553, 14.000088052430122,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            15.000044026886828, 17.00563551321351, 2.321928094887362,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            20.000001375860553, 18.00000550343433, 18.00000550343433, 12.0003521774803,
            # stride
            4.087462841250339,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Buffer B_shared
    assert_allclose(
        actual=f[75:93],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            0, 1, 0,
            # bytes, unique_bytes, lines, unique_lines
            22.00000034396526, 9.002815015607053, 20.000001375860553, 3.169925001442312,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            3.169925001442312, 9.61654884377899, 8.005624549193879,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            14.000088052430122, 1.584962500721156, 12.0003521774803, 0.044394119358453436,
            # stride
            4.087462841250339,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3 - 2.5: Dummy padding
    assert_allclose(
        actual=f[93:147],
        desired=[0] * (18 * 3),
        rtol=1e-5,
        atol=1e-5,
    )
    # TODO(@Kathryn-cat): Test group 3 - 5
    ### Check feature[2]: BufferStore(C_local) <= C_local[...] + A_shared[...] * B_shared[...]
    f = feature[2]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:57],
        desired=[
            # fmt: off
            # float math ops
            0, 27.000000010748916, 27.000000010748916, 0, 0, 0, 0,
            # int math ops
            0, 28.000000005374456, 28.000000005374456, 0, 0, 0, 0,
            # bool/select ops
            0, 0,
            # vectorize
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # unroll
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # parallel
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Buffer B_shared
    assert_allclose(
        actual=f[57:75],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            29.00000000268723, 9.002815015607053, 23.00000017198264, 3.169925001442312,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            5.044394119358453, 7.651051691178929, 5.044394119358453,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            24.000000085991324, 4.087462841250339, 18.00000550343433, 0.32192809488736235,
            # stride
            1.0,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Buffer C_local
    assert_allclose(
        actual=f[75:93],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            0, 0, 1,
            # bytes, unique_bytes, lines, unique_lines
            29.00000000268723, 11.000704269011246, 23.00000017198264, 5.044394119358453,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            0, 1, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            1.6147098441152081, 3.2094533656289497, 1,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            29.00000000268723, 11.000704269011246, 23.00000017198264, 5.044394119358453,
            # stride
            1.0,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3: Buffer A_shared
    assert_allclose(
        actual=f[93:111],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            29.00000000268723, 12.0003521774803, 19.00000275171979, 9.002815015607053,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            1.0, 3.700439718141092, 4.087462841250339,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            25.000000042995662, 8.005624549193879, 15.000044026886828, 5.044394119358453,
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
        desired=[0] * (18 * 2),
        rtol=1e-5,
        atol=1e-5,
    )
    # TODO(@Kathryn-cat): Test group 3 - 5
    ### Check feature[3]: BufferStore(C) <= C_local[...]
    f = feature[3]
    # Group 1.1: arith
    assert_allclose(
        actual=f[0:57],
        desired=[
            # fmt: off
            # float math ops
            0, 0, 0, 0, 0, 0, 0,
            # int math ops
            0, 0, 0, 0, 0, 0, 0,
            # bool/select ops
            0, 0,
            # vectorize
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # unroll
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # parallel
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            # is_gpu, blockIdx.x/y/z, threadIdx.x/y/z, vthread
            1.0, 3.169925001442312, 1.0, 1.0, 4.087462841250339, 1.0, 1.0, 2.321928094887362,
            # fmt: on
            ######
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.1: Buffer C
    assert_allclose(
        actual=f[57:75],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            0, 1, 0,
            # bytes, unique_bytes, lines, unique_lines
            20.000001375860553, 20.000001375860553, 14.000088052430122, 14.000088052430122,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            0, 0, 1,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            0, 0, 0,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            21.000000687930438, 21.000000687930438, 15.000044026886828, 15.000044026886828,
            # stride
            1.0,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.2: Buffer C_local
    assert_allclose(
        actual=f[75:93],
        desired=[
            # fmt: off
            # AccessType: read, write, read & write
            1, 0, 0,
            # bytes, unique_bytes, lines, unique_lines
            20.000001375860553, 11.000704269011246, 14.000088052430122, 5.044394119358453,
            # ReuseType: loop multiple read, serial multiple read write, no reuse
            1, 0, 0,
            # reuse_dis_iter, reuse_dis_bytes, reuse_ct
            9.002815015607053, 12.0003521774803, 4.087462841250339,
            # (byte, unique_bytes, lines, unique_lines) / reuse_ct
            16.00002201361136, 7.011227255423254, 10.001408194392809, 1.584962500721156,
            # stride
            1.0,
            # fmt: on
        ],
        rtol=1e-5,
        atol=1e-5,
    )
    # Group 2.3 - 2.5: Dummy padding
    assert_allclose(
        actual=f[93:147],
        desired=[0] * (18 * 3),
        rtol=1e-5,
        atol=1e-5,
    )
    # TODO(@Kathryn-cat): Test group 3 - 5


if __name__ == "__main__":
    test_cpu_matmul()
    test_cpu_fusion()
    test_gpu()
