import argparse
import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt

from tvm.contrib.tar import untar


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path_orig", type=str)
    parser.add_argument("--load_path_ten", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument(
        "--type",
        type=str,
        default="weight",
        choices=["weight", "gain", "cover", "total_gain", "total_cover"],
    )
    return parser.parse_args()


ARGS = _parse_args()


GROUP_1: List[str] = []
GROUP_1.extend(
    f"Group_1.{s}"
    for s in [
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
)


GROUP_2: List[str] = []
for i in range(5):
    GROUP_2.extend(
        f"Group_2.B{i}.{s}"
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


GROUP_3: List[str] = []
GROUP_3.extend(f"Group_3.arith_intensity_curve_{i}" for i in range(10))


GROUP_4_TEN: List[str] = []
GROUP_4_TEN.extend(
    f"Group_4.{s}"
    for s in [
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
    ]
)


GROUP_4_ORIG: List[str] = []
GROUP_4_ORIG.extend(
    f"Group_4.{s}"
    for s in [
        "alloc_size",
        "alloc_prod",
        "alloc_outer_prod",
        "alloc_inner_prod",
    ]
)


GROUP_5: List[str] = []
GROUP_5.extend(
    f"Group_5.{s}"
    for s in [
        "outer_prod",
        "num_loops",
        "auto_unroll_max_step",
    ]
)


assert len(GROUP_1) == 57
assert len(GROUP_2) == 90
assert len(GROUP_3) == 10
assert len(GROUP_4_ORIG) == 4
assert len(GROUP_4_TEN) == 12
assert len(GROUP_5) == 3
FEATURES_ORIG = GROUP_1 + GROUP_2 + GROUP_3 + GROUP_4_ORIG + GROUP_5
FEATURES_TEN = GROUP_1 + GROUP_2 + GROUP_3 + GROUP_4_TEN + GROUP_5
ALL_FEATURES = GROUP_1 + GROUP_2 + GROUP_3 + GROUP_4_ORIG + GROUP_4_TEN + GROUP_5


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.bin")
        data_path = os.path.join(tmp_dir, "data.npy")
        untar(ARGS.load_path_orig, tmp_dir)
        model = xgb.Booster()
        model.load_model(model_path)
    map_orig = model.get_score(importance_type=ARGS.type)
    keys = [FEATURES_ORIG[int(s[1:])] for s in list(map_orig.keys())]
    values = list(map_orig.values())
    map_orig = dict(zip(keys, values))

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.bin")
        data_path = os.path.join(tmp_dir, "data.npy")
        untar(ARGS.load_path_ten, tmp_dir)
        model = xgb.Booster()
        model.load_model(model_path)
    map_ten = model.get_score(importance_type=ARGS.type)
    keys = [FEATURES_TEN[int(s[1:])] for s in list(map_ten.keys())]
    values = list(map_ten.values())
    map_ten = dict(zip(keys, values))

    map = {}
    for key in ALL_FEATURES:
        if key in map_orig or key in map_ten:
            map[key] = [0, 0]
            if key in map_orig:
                map[key][0] = map_orig[key]
            if key in map_ten:
                map[key][1] = map_ten[key]

    map = dict(sorted(map.items(), key=lambda item: item[1][1]))
    names, scores = list(map.keys()), np.array(list(map.values()))
    score_orig = scores[:, 0]
    score_ten = scores[:, 1]
    plt.style.use("seaborn-deep")
    df = pd.DataFrame({"tensorization": score_ten, "original": score_orig}, index=names)
    df.plot.barh(figsize=(18, 12))
    plt.legend(loc="lower right")
    plt.xlabel("Importance Scores")
    plt.ylabel("Feature name")
    plt.tight_layout()
    plt.savefig(ARGS.save_path)
