# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--shape", type=int, nargs="+", required=True)
parser.add_argument("--axes", type=int, nargs="+", required=True)
args = parser.parse_args()

shape = tuple(args.shape)
axes = tuple(args.axes)

random_seed = 0
rng = np.random.default_rng(random_seed)

X = rng.random(size=shape, dtype=float)

# Calculate expected output data
X_mean = np.mean(X, axis=axes, keepdims=True)
X_std = np.std(X, axis=axes, keepdims=True)
Y = (X - X_mean) / X_std


def to_c_float_literals(arr):
    literals_per_line = 8
    literals = [f"{literal:.7f}f" for literal in arr.flatten().tolist()]
    result = ""
    for i, literal in enumerate(literals):
        result += "{},{}".format(literal, "\n" if (i + 1) % literals_per_line == 0 else " ")
    return result


print(f"input:\n{to_c_float_literals(X)}")
print(f"expected output:\n{to_c_float_literals(Y)}")
