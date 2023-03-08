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

from __future__ import annotations

import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax, runtime
from tvm.ir.transform import PassContext
from tvm.relax.library import cutlass_codegen_with_match_results, get_cutlass_pattern
from tvm.relax.transform import LegalizeOps, MetaScheduleApplyDatabase
from tvm.relax.vm import build as relax_build
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

PKG_FILE = "/tmp/test_library_cutlass_structural_info.so"
GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


def test_transform(mod):
    mod = relax.transform.Transform2GEMM()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    return mod


def dispatch_and_tune(mod, tune=True):
    mod = relax.transform.SplitCallTIRByPattern(
        get_cutlass_pattern(), cutlass_codegen_with_match_results
    )(mod)
    if tune:
        target = tvm.target.Target("nvidia/geforce-rtx-3090")
        with tempfile.TemporaryDirectory() as work_dir:
            db = ms.relax_integration.tune_relax(
                mod=mod,
                params=None,
                target=target,
                num_trials_per_iter=32,
                max_trials_global=64,
                work_dir=work_dir,
            )
        with target, db, PassContext(opt_level=3):
            new_mod = MetaScheduleApplyDatabase()(mod)
        return new_mod
    return mod


def test_build(mod, args, target_res, target="cuda", tune=True, rtol=1e-2):
    if target == "cuda":
        mod = dispatch_and_tune(mod, tune)
    executable = relax_build(mod, target)
    executable.mod.export_library(PKG_FILE, cc="nvcc")
    dev = tvm.cpu() if target == "llvm" else tvm.cuda()
    args_tvm = [tvm.nd.array(arg, dev) for arg in args]
    executable = tvm.runtime.load_module(PKG_FILE)
    vm = relax.vm.VirtualMachine(exec=executable, device=dev)
    res = vm["main"](*args_tvm)
    np.testing.assert_allclose(res.numpy(), target_res, rtol=rtol)
    print("test passed!")


# ------------------------- basic tests below ---------------------------


def constructGEMM(m, n, k):
    @tvm.script.ir_module
    class HGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

        @R.function
        def main(A: R.Tensor((m, k), A_TYPE), B: R.Tensor((k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm, (A, B), R.Tensor((m, n), C_TYPE))
                R.output(C)
            return C

    return HGEMM


@tvm.testing.requires_cutlass
def test_cutlass_dense():
    m, n, k = 32, 128, 64
    mod = constructGEMM(m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    test_build(mod, [A, B], A @ B, tune=False)


def constructBatchGEMM(b, m, n, k):
    @tvm.script.ir_module
    class BatchHGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb, l0, l1, l2 in T.grid(b, m, n, k):
                with T.block("batch_dense_row_row_row"):
                    vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                    T.reads(A[vb, vi, vk], B[vk, vj])
                    T.writes(C[vb, vi, vj])
                    with T.init():
                        C[vb, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb, vi, vj] += A[vb, vi, vk] * B[vk, vj]

        @R.function
        def main(A: R.Tensor((b, m, k), A_TYPE), B: R.Tensor((k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return BatchHGEMM


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense():
    b, m, n, k = 2, 32, 128, 64
    mod = constructBatchGEMM(b, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    test_build(mod, [A, B], A @ B)


def constructBatchGEMM2(b, m, n, k):
    @tvm.script.ir_module
    class BatchHGEMM2:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (b, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb, l0, l1, l2 in T.grid(b, m, n, k):
                with T.block("batch_dense_row_row_row"):
                    vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                    T.reads(A[vb, vi, vk], B[vb, vk, vj])
                    T.writes(C[vb, vi, vj])
                    with T.init():
                        C[vb, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb, vi, vj] += A[vb, vi, vk] * B[vb, vk, vj]

        @R.function
        def main(A: R.Tensor((b, m, k), A_TYPE), B: R.Tensor((b, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return BatchHGEMM2


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense2():
    b, m, n, k = 2, 32, 128, 64
    mod = constructBatchGEMM2(b, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    test_build(mod, [A, B], A @ B, tune=False)


# ------------------------- einsum tests below ---------------------------

# einsum "ghij, ghjk -> ghik"
def constructMultiBatchGEMM(b1, b2, m, n, k):
    @tvm.script.ir_module
    class MultiBatchHGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b1, b2, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (b1, b2, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b1, b2, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb1, lb2, l0, l1, l2 in T.grid(b1, b2, m, n, k):
                with T.block("multi_batch_dense_row_row_row"):
                    vb1, vb2, vi, vj, vk = T.axis.remap("SSSSR", [lb1, lb2, l0, l1, l2])
                    T.reads(A[vb1, vb2, vi, vk], B[vb1, vb2, vk, vj])
                    T.writes(C[vb1, vb2, vi, vj])
                    with T.init():
                        C[vb1, vb2, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb1, vb2, vi, vj] += A[vb1, vb2, vi, vk] * B[vb1, vb2, vk, vj]

        @R.function
        def main(A: R.Tensor((b1, b2, m, k), A_TYPE), B: R.Tensor((b1, b2, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b1, b2, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b1, b2, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return MultiBatchHGEMM


@tvm.testing.requires_cutlass
def test_cutlass_multi_batch_dense():
    b1, b2, m, n, k = 2, 3, 32, 128, 64
    mod = constructMultiBatchGEMM(b1, b2, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, b2, k, n).astype("float16") * 5
    test_build(mod, [A, B], A @ B)


# einsum "ghij, hjk -> ghik"
def constructMultiBatchGEMM2(b1, b2, m, n, k):
    @tvm.script.ir_module
    class MultiBatchHGEMM2:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b1, b2, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (b2, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b1, b2, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb1, lb2, l0, l1, l2 in T.grid(b1, b2, m, n, k):
                with T.block("multi_batch_dense_row_row_row"):
                    vb1, vb2, vi, vj, vk = T.axis.remap("SSSSR", [lb1, lb2, l0, l1, l2])
                    T.reads(A[vb1, vb2, vi, vk], B[vb2, vk, vj])
                    T.writes(C[vb1, vb2, vi, vj])
                    with T.init():
                        C[vb1, vb2, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb1, vb2, vi, vj] += A[vb1, vb2, vi, vk] * B[vb2, vk, vj]

        @R.function
        def main(A: R.Tensor((b1, b2, m, k), A_TYPE), B: R.Tensor((b2, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b1, b2, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b1, b2, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return MultiBatchHGEMM2


@tvm.testing.requires_cutlass
def test_cutlass_multi_batch_dense2():
    b1, b2, m, n, k = 2, 3, 32, 128, 64
    mod = constructMultiBatchGEMM2(b1, b2, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b2, k, n).astype("float16") * 5
    test_build(mod, [A, B], A @ B)


# einsum "ghij, gjk -> ghik"
def constructMultiBatchGEMM3(b1, b2, m, n, k):
    @tvm.script.ir_module
    class MultiBatchHGEMM3:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b1, b2, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (b1, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b1, b2, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb1, lb2, l0, l1, l2 in T.grid(b1, b2, m, n, k):
                with T.block("multi_batch_dense_row_row_row"):
                    vb1, vb2, vi, vj, vk = T.axis.remap("SSSSR", [lb1, lb2, l0, l1, l2])
                    T.reads(A[vb1, vb2, vi, vk], B[vb1, vk, vj])
                    T.writes(C[vb1, vb2, vi, vj])
                    with T.init():
                        C[vb1, vb2, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb1, vb2, vi, vj] += A[vb1, vb2, vi, vk] * B[vb1, vk, vj]

        @R.function
        def main(A: R.Tensor((b1, b2, m, k), A_TYPE), B: R.Tensor((b1, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b1, b2, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b1, b2, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return MultiBatchHGEMM3


@tvm.testing.requires_cutlass
def test_cutlass_multi_batch_dense3():
    b1, b2, m, n, k = 2, 3, 32, 128, 64
    mod = constructMultiBatchGEMM3(b1, b2, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, k, n).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("ghij,gjk->ghik", A, B))


# einsum "ij, ik -> jk"
def constructTransGEMM(m, n, k):
    @tvm.script.ir_module
    class TransHGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (k, m), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vk, vi], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vk, vi] * B[vk, vj]

        @R.function
        def main(A: R.Tensor((k, m), A_TYPE), B: R.Tensor((k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm, (A, B), R.Tensor((m, n), C_TYPE))
                R.output(C)
            return C

    return TransHGEMM


@tvm.testing.requires_cutlass
def test_cutlass_trans_dense():
    m, n, k = 32, 128, 64
    mod = constructTransGEMM(m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(k, m).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    test_build(mod, [A, B], A.T @ B)


# einsum "ik, jk -> ij"
def constructTransGEMM2(m, n, k):
    @tvm.script.ir_module
    class TransHGEMM2:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (n, k), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("trans_dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vj, vk])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vi, vk] * B[vj, vk]

        @R.function
        def main(A: R.Tensor((m, k), A_TYPE), B: R.Tensor((n, k), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm, (A, B), R.Tensor((m, n), C_TYPE))
                R.output(C)
            return C

    return TransHGEMM2


@tvm.testing.requires_cutlass
def test_cutlass_trans_dense2():
    m, n, k = 32, 128, 64
    mod = constructTransGEMM2(m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(n, k).astype("float16") * 5
    test_build(mod, [A, B], A @ B.T)


# einsum "hij, hjk -> ik"
def constructReductionGEMM(b, m, n, k):
    @tvm.script.ir_module
    class ReductionHGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (b, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb, l0, l1, l2 in T.grid(b, m, n, k):
                with T.block("reduction_dense_row_row_row"):
                    vb, vi, vj, vk = T.axis.remap("RSSR", [lb, l0, l1, l2])
                    T.reads(A[vb, vi, vk], B[vb, vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vb, vi, vk] * B[vb, vk, vj]

        @R.function
        def main(A: R.Tensor((b, m, k), A_TYPE), B: R.Tensor((b, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm, (A, B), R.Tensor((m, n), C_TYPE))
                R.output(C)
            return C

    return ReductionHGEMM


@tvm.testing.requires_cutlass
def test_cutlass_reduction_dense():
    b, m, n, k = 2, 32, 128, 64
    mod = constructReductionGEMM(b, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("hij,hjk->ik", A, B))


# einsum "ghij, jhk -> gik"
def constructReductionGEMM2(b1, b2, m, n, k):
    @tvm.script.ir_module
    class ReductionHGEMM2:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b1, b2, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, b2, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b1, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb1, lb2, l0, l1, l2 in T.grid(b1, b2, m, n, k):
                with T.block("reduction_dense_row_row_row"):
                    vb1, vb2, vi, vj, vk = T.axis.remap("SRSSR", [lb1, lb2, l0, l1, l2])
                    T.reads(A[vb1, vb2, vi, vk], B[vk, vb2, vj])
                    T.writes(C[vb1, vi, vj])
                    with T.init():
                        C[vb1, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb1, vi, vj] += A[vb1, vb2, vi, vk] * B[vk, vb2, vj]

        @R.function
        def main(A: R.Tensor((b1, b2, m, k), A_TYPE), B: R.Tensor((k, b2, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b1, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b1, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return ReductionHGEMM2


@tvm.testing.requires_cutlass
def test_cutlass_reduction_dense2():
    b1, b2, m, n, k = 2, 3, 32, 128, 64
    mod = constructReductionGEMM2(b1, b2, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(k, b2, n).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("ghij,jhk->gik", A, B))


# einsum "ghij, gjk -> hik"
def constructReductionGEMM3(b1, b2, m, n, k):
    @tvm.script.ir_module
    class ReductionHGEMM3:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b1, b2, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (b1, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b2, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb1, lb2, l0, l1, l2 in T.grid(b1, b2, m, n, k):
                with T.block("reduction_dense_row_row_row"):
                    vb1, vb2, vi, vj, vk = T.axis.remap("RSSSR", [lb1, lb2, l0, l1, l2])
                    T.reads(A[vb1, vb2, vi, vk], B[vb1, vk, vj])
                    T.writes(C[vb2, vi, vj])
                    with T.init():
                        C[vb2, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb2, vi, vj] += A[vb1, vb2, vi, vk] * B[vb1, vk, vj]

        @R.function
        def main(A: R.Tensor((b1, b2, m, k), A_TYPE), B: R.Tensor((b1, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b2, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b2, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return ReductionHGEMM3


@tvm.testing.requires_cutlass
def test_cutlass_reduction_dense3():
    b1, b2, m, n, k = 2, 3, 32, 128, 64
    mod = constructReductionGEMM3(b1, b2, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, k, n).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("ghij,gjk->hik", A, B))


# einsum "ghij, ghjk -> ik"
def constructReductionGEMM4(b1, b2, m, n, k):
    @tvm.script.ir_module
    class ReductionHGEMM4:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b1, b2, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (b1, b2, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb1, lb2, l0, l1, l2 in T.grid(b1, b2, m, n, k):
                with T.block("reduction_dense_row_row_row"):
                    vb1, vb2, vi, vj, vk = T.axis.remap("RRSSR", [lb1, lb2, l0, l1, l2])
                    T.reads(A[vb1, vb2, vi, vk], B[vb1, vb2, vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vb1, vb2, vi, vk] * B[vb1, vb2, vk, vj]

        @R.function
        def main(A: R.Tensor((b1, b2, m, k), A_TYPE), B: R.Tensor((b1, b2, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm, (A, B), R.Tensor((m, n), C_TYPE))
                R.output(C)
            return C

    return ReductionHGEMM4


@tvm.testing.requires_cutlass
def test_cutlass_reduction_dense4():
    b1, b2, m, n, k = 2, 3, 32, 128, 64
    mod = constructReductionGEMM4(b1, b2, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b1, b2, m, k).astype("float16") * 5
    B = np.random.rand(b1, b2, k, n).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("ghij,ghjk->ik", A, B), rtol=15)


# einsum "hij, ijk -> hik"
def constructPermutationGEMM(b, m, n, k):
    @tvm.script.ir_module
    class PermutationHGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (m, k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb, l0, l1, l2 in T.grid(b, m, n, k):
                with T.block("permutation_dense_row_row_row"):
                    vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                    T.reads(A[vb, vi, vk], B[vi, vk, vj])
                    T.writes(C[vb, vi, vj])
                    with T.init():
                        C[vb, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb, vi, vj] += A[vb, vi, vk] * B[vi, vk, vj]

        @R.function
        def main(A: R.Tensor((b, m, k), A_TYPE), B: R.Tensor((m, k, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return PermutationHGEMM


@tvm.testing.requires_cutlass
def test_cutlass_permutation_dense():
    b, m, n, k = 2, 32, 128, 64
    mod = constructPermutationGEMM(b, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(m, k, n).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("hij,ijk->hik", A, B))


# einsum "hij, kj -> hik"
def constructPermutationGEMM2(b, m, n, k):
    @tvm.script.ir_module
    class PermutationHGEMM2:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b, m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (n, k), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb, l0, l1, l2 in T.grid(b, m, n, k):
                with T.block("permutation_dense_row_row_row"):
                    vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                    T.reads(A[vb, vi, vk], B[vj, vk])
                    T.writes(C[vb, vi, vj])
                    with T.init():
                        C[vb, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb, vi, vj] += A[vb, vi, vk] * B[vj, vk]

        @R.function
        def main(A: R.Tensor((b, m, k), A_TYPE), B: R.Tensor((n, k), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return PermutationHGEMM2


@tvm.testing.requires_cutlass
def test_cutlass_permutation_dense2():
    b, m, n, k = 2, 32, 128, 64
    mod = constructPermutationGEMM2(b, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(n, k).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("hij,kj->hik", A, B))


# einsum "hij, ikj -> hkj"
def constructPermutationGEMM3(b, m, n, k):
    @tvm.script.ir_module
    class PermutationHGEMM3:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (b, k, n), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, m, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (b, m, n), C_TYPE)  # pylint: disable=invalid-name
            for lb, l0, l1, l2 in T.grid(b, m, n, k):
                with T.block("permutation_dense_row_row_row"):
                    vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                    T.reads(A[vb, vk, vj], B[vk, vi, vj])
                    T.writes(C[vb, vi, vj])
                    with T.init():
                        C[vb, vi, vj] = T.cast(0.0, C_TYPE)
                    C[vb, vi, vj] += A[vb, vk, vj] * B[vk, vi, vj]

        @R.function
        def main(A: R.Tensor((b, k, n), A_TYPE), B: R.Tensor((k, m, n), B_TYPE)):
            with R.dataflow():
                C: R.Tensor((b, m, n), C_TYPE) = R.call_tir(
                    hgemm, (A, B), R.Tensor((b, m, n), C_TYPE)
                )
                R.output(C)
            return C

    return PermutationHGEMM3


@tvm.testing.requires_cutlass
def test_cutlass_permutation_dense3():
    b, m, n, k = 2, 32, 128, 64
    mod = constructPermutationGEMM3(b, m, n, k)
    mod = test_transform(mod)
    A = np.random.rand(b, k, n).astype("float16") * 5
    B = np.random.rand(k, m, n).astype("float16") * 5
    test_build(mod, [A, B], np.einsum("hij,ikj->hkj", A, B))


# ------------------------- multiple einsum tests below ---------------------------


def constructManyGEMM(m, n, k, p):
    @tvm.script.ir_module
    class ManyHGEMM:
        @T.prim_func
        def hgemm1(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

        @T.prim_func
        def hgemm2(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, n), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (n, p), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, p), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, p, n):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            A: R.Tensor((m, k), A_TYPE), B: R.Tensor((k, n), B_TYPE), D: R.Tensor((n, p), B_TYPE)
        ):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm1, (A, B), R.Tensor((m, n), C_TYPE))
                res: R.Tensor((m, p), C_TYPE) = R.call_tir(hgemm2, (C, D), R.Tensor((m, p), C_TYPE))
                R.output(res)
            return res

    return ManyHGEMM


@tvm.testing.requires_cutlass
def test_cutlass_many_dense():
    m, n, k, p = 32, 128, 64, 256
    mod = constructManyGEMM(m, n, k, p)
    mod = test_transform(mod)
    A = np.random.rand(m, k).astype("float16") * 3
    B = np.random.rand(k, n).astype("float16") * 3
    C = np.random.rand(n, p).astype("float16") * 3
    test_build(mod, [A, B, C], (A @ B) @ C, tune=False)


def constructManyGEMM2(m, n, k, p):
    @tvm.script.ir_module
    class ManyHGEMM2:
        @T.prim_func
        def hgemm1(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, k), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

        @T.prim_func
        def hgemm2(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, n), A_TYPE)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (p, n), B_TYPE)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, p), C_TYPE)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, p, n):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vj, vk])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, C_TYPE)
                    C[vi, vj] += A[vi, vk] * B[vj, vk]

        @R.function
        def main(
            A: R.Tensor((m, k), A_TYPE), B: R.Tensor((k, n), B_TYPE), D: R.Tensor((p, n), B_TYPE)
        ):
            with R.dataflow():
                C: R.Tensor((m, n), C_TYPE) = R.call_tir(hgemm1, (A, B), R.Tensor((m, n), C_TYPE))
                res: R.Tensor((m, p), C_TYPE) = R.call_tir(hgemm2, (C, D), R.Tensor((m, p), C_TYPE))
                R.output(res)
            return res

    return ManyHGEMM2


@tvm.testing.requires_cutlass
def test_cutlass_many_dense2():
    m, n, k, p = 32, 128, 64, 256
    mod = constructManyGEMM2(m, n, k, p)
    mod = test_transform(mod)
    A = np.random.rand(m, k).astype("float16") * 3
    B = np.random.rand(k, n).astype("float16") * 3
    C = np.random.rand(p, n).astype("float16") * 3
    test_build(mod, [A, B, C], (A @ B) @ C.T)


if __name__ == "__main__":
    test_cutlass_dense()
    test_cutlass_batch_dense()
    test_cutlass_batch_dense2()
    test_cutlass_multi_batch_dense()
    test_cutlass_multi_batch_dense2()
    test_cutlass_multi_batch_dense3()
    test_cutlass_trans_dense()
    test_cutlass_trans_dense2()
    test_cutlass_reduction_dense()
    test_cutlass_reduction_dense2()
    test_cutlass_reduction_dense3()
    test_cutlass_reduction_dense4()
    test_cutlass_permutation_dense()
    test_cutlass_permutation_dense2()
    test_cutlass_permutation_dense3()
    test_cutlass_many_dense()
    test_cutlass_many_dense2()
