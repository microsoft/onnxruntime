# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
import sys
from dataclasses import dataclass
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest
from utils import dtype_to_bytes


def get_bert_sizes():
    batch_sizes = [1]
    seq_lens = [384]
    hidden_sizes = [1024]
    return product(batch_sizes, seq_lens, hidden_sizes)


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: re.match("FastGelu.*_half.*", x), dir(ke))),
        "float32": list(filter(lambda x: re.match("FastGelu.*_float.*", x), dir(ke))),
        "float64": list(filter(lambda x: re.match("FastGelu.*_double.*", x), dir(ke))),
    }
    return type_map[dtype]


def fast_gelu(x, bias):
    x = x + bias
    y = 0.5 * x * (1 + np.tanh(0.797885 * x + 0.035677 * x * x * x))
    return y


def run_fast_gelu(x_size, bias_size, dtype, func):
    np.random.seed(0)
    x = np.random.rand(*x_size).astype(dtype)
    bias = np.random.rand(bias_size).astype(dtype)
    y = np.random.rand(*x_size).astype(dtype)

    x_d = ke.DeviceArray(x)
    bias_d = ke.DeviceArray(bias)
    y_d = ke.DeviceArray(y)
    f = getattr(ke, func)
    my_op = f(x_d, bias_d, y_d, x.size, bias.size)
    if my_op.IsSupported():
        my_op.Run()
        y_d.UpdateHostNumpyArray()

        y_ref = fast_gelu(x, bias)
        np.testing.assert_allclose(y_ref, y, rtol=1e-02)


test_cases = [((2, 16), 16), ((1, 2, 768), 768), ((1, 2, 1024), 1024), ((1, 3, 3), 3)]
dtypes = ["float16", "float32", "float64"]


@pytest.mark.parametrize("x_size, bias_size", test_cases)
@pytest.mark.parametrize("dtype", dtypes)
def test_fast_gelu(x_size, bias_size, dtype):
    for f in dtype_to_funcs(dtype):
        run_fast_gelu(x_size, bias_size, dtype, f)


@dataclass
class FastGeluMetric(ke.BandwidthMetric):
    batch_size: int
    seq_len: int
    hidden_size: int

    def report(self):
        prefix = f"{self.name:<50} {self.dtype}  batch_size={self.batch_size:<4} seq_len={self.seq_len:<4} hidden_size={self.hidden_size:<4} "
        if self.duration > 0:
            return prefix + f"{self.duration:.2f} us, {self.gbps:.2f} GB/s"
        return prefix + "not supported or redundant"


def profile_fast_gelu_func(batch_size, seq_len, hidden_size, dtype, func):
    x_size = [batch_size, seq_len, hidden_size * 4]
    bias_size = hidden_size * 4
    np.random.seed(0)
    x = np.random.rand(*x_size).astype(dtype)
    bias = np.random.rand(bias_size).astype(dtype)
    y = np.random.rand(*x_size).astype(dtype)

    x_d = ke.DeviceArray(x)
    bias_d = ke.DeviceArray(bias)
    y_d = ke.DeviceArray(y)
    f = getattr(ke, func)
    my_op = f(x_d, bias_d, y_d, x.size, bias.size)

    duration_ms = -1
    if my_op.IsSupported():
        duration_ms = my_op.Profile()
    total_bytes = (x.size * 2 + bias.size) * dtype_to_bytes(dtype)

    ke.report(FastGeluMetric(func, dtype, duration_ms, total_bytes, batch_size, seq_len, hidden_size))


def profile_with_args(batch_size, seq_len, hidden_size, dtype, sort):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(dtype):
            profile_fast_gelu_func(batch_size, seq_len, hidden_size, dtype, func)


def profile():
    for dtype in dtypes:
        for bert_size in get_bert_sizes():
            profile_with_args(*bert_size, dtype, True)
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("batch_size", type=int)
    group.add_argument("seq_len", type=int)
    group.add_argument("hidden_size", type=int)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.batch_size, args.seq_len, args.hidden_size, args.dtype, args.sort)
