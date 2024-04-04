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
from utils import dtype_to_bytes, fast_gelu, gelu, relu


def get_bert_sizes():
    batch_sizes = [1]
    seq_lens = [384]
    hidden_sizes = [1024]
    return product(batch_sizes, seq_lens, hidden_sizes)


def dtype_to_funcs(fn_name, dtype):
    type_map = {
        "float16": list(filter(lambda x: re.match(f"{fn_name}.*_half.*", x), dir(ke))),
        "float32": list(filter(lambda x: re.match(f"{fn_name}.*_float.*", x), dir(ke))),
        "float64": list(filter(lambda x: re.match(f"{fn_name}.*_double.*", x), dir(ke))),
    }
    return type_map[dtype]


def fn_name_to_ref_impl(fn_name):
    return {
        "FastGeLU": fast_gelu,
        "GeLU": gelu,
        "ReLU": relu,
    }[fn_name]


def run_elementwise(x_size, bias_size, fn_name, dtype, func):
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

        ref_fn = fn_name_to_ref_impl(fn_name)
        y_ref = ref_fn(x, bias)
        np.testing.assert_allclose(y_ref, y, atol=1e-3, rtol=1e-04)


test_cases = [((2, 16), 16), ((1, 2, 768), 768), ((1, 2, 1024), 1024), ((1, 2, 1027), 1027), ((1, 3, 3), 3)]
fn_names = ["FastGeLU", "GeLU", "ReLU"]
dtypes = ["float16", "float32"]


@pytest.mark.parametrize("x_size, bias_size", test_cases)
@pytest.mark.parametrize("dtype", dtypes)
def test_fast_gelu(x_size, bias_size, dtype):
    for f in dtype_to_funcs("FastGeLU", dtype):
        run_elementwise(x_size, bias_size, "FastGeLU", dtype, f)


@pytest.mark.parametrize("fn_name", fn_names)
@pytest.mark.parametrize("dtype", dtypes)
def test_elementwise_fns(fn_name, dtype):
    for f in dtype_to_funcs(fn_name, dtype):
        run_elementwise((1, 2, 768), 768, fn_name, dtype, f)


@dataclass
class ElementwiseMetric(ke.BandwidthMetric):
    batch_size: int
    seq_len: int
    hidden_size: int

    def report(self):
        common = f"{self.dtype}  batch_size={self.batch_size:<4} seq_len={self.seq_len:<4} hidden_size={self.hidden_size:<4} {self.name}"
        if self.duration > 0:
            return f"{self.duration:>6.2f} us {self.gbps:>5.2f} GB/s " + common
        return "not supported        " + common


def profile_elementwise_func(batch_size, seq_len, hidden_size, dtype, func):
    x_size = [batch_size, seq_len, hidden_size]
    bias_size = hidden_size
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

    ke.report(ElementwiseMetric(func, dtype, duration_ms, total_bytes, batch_size, seq_len, hidden_size))


def profile_with_args(batch_size, seq_len, hidden_size, fn_name, dtype, sort):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(fn_name, dtype):
            profile_elementwise_func(batch_size, seq_len, hidden_size, dtype, func)


def profile():
    for dtype in dtypes:
        for bert_size in get_bert_sizes():
            profile_with_args(*bert_size, "FastGeLU", dtype, True)
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("batch_size", type=int)
    group.add_argument("seq_len", type=int)
    group.add_argument("hidden_size", type=int)
    group.add_argument("fn_name", choices=fn_names)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.batch_size, args.seq_len, args.hidden_size, args.fn_name, args.dtype, args.sort)
