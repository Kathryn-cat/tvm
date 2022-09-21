import numpy as np

import tvm
from tvm.script import tir as T


# find shape of input automatically
@T.prim_func
def vector_add(a: T.handle, b: T.handle, c: T.handle) -> None:
    m = T.var("int32")
    A = T.match_buffer(a, (m, ), "float32")
    B = T.match_buffer(b, (m, ), "float32")
    C = T.match_buffer(c, (m, ), "float32")
    for i in T.grid(m):
        with T.block("update"):
            vi = T.axis.remap("S", [i])
            C[vi] = A[vi] + B[vi]


def evaluate(mod: tvm.runtime.Module, m: int):
    a = np.random.rand(m).astype("float32")
    b = np.random.rand(m).astype("float32")
    c = np.random.rand(m).astype("float32")

    a = tvm.nd.array(a)
    b = tvm.nd.array(b)
    c = tvm.nd.array(c)
    mod(a, b, c)
    c = c.numpy()
    print(c)



def main():
    target = tvm.target.Target("llvm")
    mod = tvm.build(vector_add, target)
    evaluate(mod, m=1024)


if __name__ == "__main__":
    main()
