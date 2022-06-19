"""
script for upgrading the database
"""

import argparse
import json
import glob
import os
import shutil
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-tuning-record",
        type=str,
        help="path to the json tuning record file",
        required=True,
    )
    parser.add_argument(
        "--updated-json-tuning-record",
        type=str,
        help="path to the new json tuning record file after upgrade",
        required=True,
    )
    return parser.parse_args()


ARGS = _parse_args()


# pylint: disable=missing-function-docstring
def upgrade(old_record_path, new_record_path):
    with open(old_record_path, "r", encoding="utf-8") as i_f:
        lines = [json.loads(line) for line in i_f]
    for line in lines:
        for inst in line[1][0][0]:
            if inst[0] in ["Split", "Fuse"]:
                inst[2] = [1]
    with open(new_record_path, "w", encoding="utf-8") as o_f:
        for line in lines:
            o_f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    os.makedirs(ARGS.updated_json_tuning_record, exist_ok=True)
    model_dirs = glob.glob(os.path.join(ARGS.json_tuning_record, "*"))
    for model_dir in tqdm(model_dirs):
        model_name = model_dir.split('/')[-1]
        new_model_dir = os.path.join(ARGS.updated_json_tuning_record, model_name)
        os.makedirs(new_model_dir, exist_ok=True)
        all_json = glob.glob(os.path.join(model_dir, "*.json"))
        for file_json in all_json:
            file_name = file_json.split('/')[-1]
            if 'workload' in file_name:
                shutil.copyfile(file_json, os.path.join(new_model_dir, file_name))
            else:
                upgrade(file_json, os.path.join(new_model_dir, file_name))
