import argparse
import os
import tempfile

import xgboost as xgb

from tvm.contrib.tar import untar


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    return parser.parse_args()


ARGS = _parse_args()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.bin")
        data_path = os.path.join(tmp_dir, "data.npy")
        untar(ARGS.path, tmp_dir)
        model = xgb.Booster()
        model.load_model(model_path)
    print(f"important feature by weight: {model.get_score()}")
    print(f'important feature by gain: {model.get_score(importance_type="gain")}')
