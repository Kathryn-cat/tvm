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
from tvm.relax.vm import build as relax_build
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_transform(mod):
    mod = relax.transform.LegalizeOps()(mod)
    return mod


# ------------------------- attention tests below ---------------------------


def constructGEMM(m, n, k, dtype="float16"):
    @tvm.script.ir_module
    class HGEMM:
        @T.prim_func
        def hgemm(x: T.handle, y: T.handle, z: T.handle) -> None:
            A = T.match_buffer(x, (m, k), dtype)  # pylint: disable=invalid-name
            B = T.match_buffer(y, (k, n), dtype)  # pylint: disable=invalid-name
            C = T.match_buffer(z, (m, n), dtype)  # pylint: disable=invalid-name
            for l0, l1, l2 in T.grid(m, n, k):
                with T.block("dense_row_row_row"):
                    vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, dtype)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

        @R.function
        def main(A: R.Tensor((m, k), dtype), B: R.Tensor((k, n), dtype)):
            with R.dataflow():
                C: R.Tensor((m, n), dtype) = R.matmul(A, B, dtype)
                R.output(C)
            return C

    return HGEMM


@tvm.testing.requires_cutlass
def test_cutlass_attention():
    m, n, k = 32, 128, 64
    mod = constructGEMM(m, n, k)
    mod.show()


if __name__ == "__main__":
    test_cutlass_attention()
