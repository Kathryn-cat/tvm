from dyn_search_space import *

if __name__ == "__main__":
    '''
    print(f'original vectoradd_const')
    print(vectoradd_const.script())
    sch = tvm.tir.Schedule(vectoradd_const)
    search_space_vectoradd_const(sch)
    print(f'new module')
    print(sch.mod.script())
    vectoradd_const_mod = tvm.build(sch.mod, target="cuda")
    '''

    print(f'original vectoradd')
    print(vectoradd.script())
    sch = tvm.tir.Schedule(vectoradd)
    search_space_vectoradd(sch)
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

    print(f"original matmul")
    print(matmul.script())
    sch = tvm.tir.Schedule(matmul)
    search_space_matmul(sch)
    print(f'new module')
    print(sch.mod.script())
    matmul_mod = tvm.build(sch.mod, target="cuda")
