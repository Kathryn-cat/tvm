import argparse

import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


ARGS = _parse_args()


if __name__ == "__main__":
    data = genfromtxt(ARGS.load_path, delimiter=",", skip_header=2)
    num_rounds, num_runs = data.shape[0], data.shape[1] // 2
    x = np.linspace(1, num_rounds, num=num_rounds)
    y_orig = np.mean(data[:, :num_runs], axis=1)
    y_ten = np.mean(data[:, num_runs:], axis=1)
    print(f"original average: {y_orig}")
    print(f"tensorization average: {y_ten}")
    y_orig_std = np.std(data[:, :num_runs], axis=1)
    y_ten_std = np.std(data[:, num_runs:], axis=1)
    plt.plot(x, y_orig, "g-", label="original")
    plt.plot(x, y_ten, "b-", label="tensorization")
    plt.fill_between(x, y_orig - y_orig_std, y_orig + y_orig_std, color="g", alpha=0.25)
    plt.fill_between(x, y_ten - y_ten_std, y_ten + y_ten_std, color="b", alpha=0.25)
    plt.xlabel("# rounds")
    plt.ylabel("GFLOPS")
    plt.legend(loc="lower right")
    plt.savefig(ARGS.save_path)

    diff = (y_ten - y_orig) / y_orig
    print(f"diff: {diff}")
