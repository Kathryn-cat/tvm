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
import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm.script import tir as T


@tvm.testing.requires_rocm
def test_rocm_inf_nan():
    def check_inf_nan(dev, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")
        sch = tvm.tir.Schedule(te.create_prim_func([A, C]))
        xo, xi = sch.split(sch.get_loops("C")[0], factors=[None, 128])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "threadIdx.x")
        fun = tvm.compile(sch.mod, "rocm")
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.rocm(0)

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float64")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float64")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float64")


@tvm.testing.requires_rocm
def test_rocm_copy():
    def check_rocm(dtype, n):
        A = te.placeholder((n,), name="A", dtype=dtype)
        dev = tvm.rocm(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(a_np)
        b_np = a.numpy()
        tvm.testing.assert_allclose(a_np, b_np)
        tvm.testing.assert_allclose(a_np, a.numpy())

    for _ in range(100):
        dtype = np.random.choice(["float32", "float16", "int8", "int32"])
        logN = np.random.randint(1, 15)
        peturb = np.random.uniform(low=0.5, high=1.5)
        check_rocm(dtype, int(peturb * (2**logN)))


@tvm.testing.requires_rocm
def test_rocm_vectorize_add():
    num_thread = 8

    def check_rocm(dtype, n, lanes):
        A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")
        sch = tir.Schedule(te.create_prim_func([A, B]))
        xo, xi = sch.split(sch.get_loops("B")[0], factors=[None, 4])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "threadIdx.x")
        fun = tvm.compile(sch.mod, target="rocm")

        dev = tvm.rocm(0)
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, dev)
        fun(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

    check_rocm("float32", 64, 2)
    check_rocm("float16", 64, 2)


@tvm.testing.requires_rocm
def test_rocm_warp_shuffle():
    @T.prim_func
    def func(
        A_handle: T.handle,
    ):
        A = T.match_buffer(A_handle, (32,), dtype="float32")

        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("test"):
                    A_local = T.alloc_buffer((1,), "float32", scope="local")
                    mask = T.alloc_buffer((1,), "uint32", scope="local")
                    t0 = T.alloc_buffer((1,), "float32", scope="local")

                    A_local[0] = A[tx]
                    A_local[0] = T.tvm_warp_shuffle(mask[0], A_local[0], 0, 32, 32)
                    A[tx] = A_local[0]

    mod = tvm.compile(func, target="rocm")
    dev = tvm.rocm(0)
    a = tvm.nd.array(np.random.uniform(size=(32,)).astype("float32"), dev)
    mod(a)
    tvm.testing.assert_allclose(a.numpy(), np.ones((32,)) * a.numpy()[0])


@tvm.testing.requires_rocm
def test_rocm_vectorized_exp():
    @T.prim_func
    def func(
        A_handle: T.handle,
        B_handle: T.handle,
    ):
        A = T.match_buffer(A_handle, (4,), dtype="float32")
        B = T.match_buffer(B_handle, (4,), dtype="float32")

        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(1, thread="threadIdx.x"):
                with T.block("test"):
                    for i in T.vectorized(0, 4):
                        B[i] = T.exp2(A[i])

    mod = tvm.compile(func, target="rocm")
    dev = tvm.rocm(0)
    a = tvm.nd.array(np.ones((4,)).astype("float32"), dev)
    b = tvm.nd.array(np.zeros((4,)).astype("float32"), dev)
    mod(a, b)
    tvm.testing.assert_allclose(b.numpy(), np.exp2(a.numpy()))
