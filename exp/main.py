from dyn_search_space import *

if __name__ == "__main__":
    print(f'original vectoradd_const')
    print(vectoradd_const.script())
    sch = tvm.tir.Schedule(vectoradd_const)
    search_space_vectoradd_const(sch)
    print(f'new module')
    print(sch.mod.script())
    vectoradd_const_mod = tvm.build(sch.mod, target="cuda")
