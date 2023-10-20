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
from utils import dtype_to_bytes, root_mean_square, standardization


def get_bert_sizes_test():
    batch_sizes = [1, 8]
    seq_lens = [64, 256]
    hidden_sizes = [1, 2, 3, 4, 5, 7, 8, 9, 13, 32, 63, 64, 65, 127, 128, 129, 177, 256, 1023, 1024]
    return product(batch_sizes, seq_lens, hidden_sizes)


def get_bert_sizes_profile():
    batch_sizes = [1, 8, 128, 256]
    seq_lens = [64, 128, 256, 384]
    hidden_sizes = [768, 1024]
    return product(batch_sizes, seq_lens, hidden_sizes)


def dtype_to_funcs(dtype, simplified=False):
    skip_layer_norm_prefix = "SimplifiedSkipLayerNorm" if simplified else "SkipLayerNorm"
    type_map = {
        "float16": list(filter(lambda x: re.match(f"{skip_layer_norm_prefix}.*_half.*", x), dir(ke))),
        "float32": list(filter(lambda x: re.match(f"{skip_layer_norm_prefix}.*_float.*", x), dir(ke))),
    }
    return type_map[dtype]


def skip_layer_norm(input_x, skip, bias, gamma, beta, epsilon):
    val = input_x + skip + bias
    output = standardization(val, 2, epsilon)
    output = output * gamma + beta
    return output, val


def simplified_skip_layer_norm(input_x, skip, bias, gamma, epsilon):
    val = input_x + skip + bias
    rms = root_mean_square(val, 2, epsilon)
    output = (val / rms) * gamma
    return output, val


def run_skip_layer_norm(
    batch_size: int, seq_len: int, hidden_size: int, dtype: str, func, simplified=False, has_optional_output=False
):
    np.random.seed(0)
    input_x = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    skip = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    bias = np.random.rand(hidden_size).astype(dtype)
    gamma = np.random.rand(hidden_size).astype(dtype)
    beta = np.random.rand(hidden_size).astype(dtype)
    # Because of rocm FMAs calculation issue with float16, epsilon should be larger when hidden_size is small
    epsilon = 0.05 if hidden_size < 8 else 0.0005
    output_y = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    output_optional = (
        np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
        if has_optional_output
        else np.empty((0), dtype=dtype)
    )

    input_d = ke.DeviceArray(input_x)
    skip_d = ke.DeviceArray(skip)
    bias_d = ke.DeviceArray(bias)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    y_d = ke.DeviceArray(output_y)
    optional_d = ke.DeviceArray(output_optional)
    f = getattr(ke, func)

    my_op = f(
        y_d,
        optional_d,
        input_d,
        skip_d,
        gamma_d,
        beta_d,
        bias_d,
        epsilon,
        hidden_size,
        batch_size * seq_len * hidden_size,
    )
    if my_op.IsSupported():
        my_op.Run()

        y_d.UpdateHostNumpyArray()
        optional_d.UpdateHostNumpyArray()

        if simplified:
            y_ref, y_optional = simplified_skip_layer_norm(input_x, skip, bias, gamma, epsilon)
        else:
            y_ref, y_optional = skip_layer_norm(input_x, skip, bias, gamma, beta, epsilon)
        np.testing.assert_almost_equal(y_ref, output_y, decimal=1)
        if has_optional_output:
            np.testing.assert_almost_equal(y_optional, output_optional, decimal=3)


dtypes = ["float32", "float16"]
simplified = [True, False]


@pytest.mark.parametrize("bert_sizes", get_bert_sizes_test())
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("simplified", simplified)
def test_skip_layer_norm(bert_sizes, dtype, simplified):
    for func in dtype_to_funcs(dtype, simplified):
        run_skip_layer_norm(*bert_sizes, dtype, func, simplified)


@dataclass
class SkipLayerNormMetric(ke.BandwidthMetric):
    batch_size: int
    seq_len: int
    hidden_size: int

    def report(self):
        common = f"{self.dtype}  batch_size={self.batch_size:<4} seq_len={self.seq_len:<4} hidden_size={self.hidden_size:<4} {self.name}"
        if self.duration > 0:
            return f"{self.duration:6.2f} us, {self.gbps:5.2f} GB/s " + common
        return "not supported          " + common


def profile_skip_layer_norm_func(batch_size, seq_len, hidden_size, dtype, func, has_optional_output):
    np.random.seed(0)
    input_x = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    skip = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    gamma = np.random.rand(hidden_size).astype(dtype)
    beta = np.random.rand(hidden_size).astype(dtype)
    bias = np.random.rand(hidden_size).astype(dtype)
    epsilon = 0.0005
    output_y = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    output_optional = (
        np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
        if has_optional_output
        else np.empty((0), dtype=dtype)
    )

    input_d = ke.DeviceArray(input_x)
    skip_d = ke.DeviceArray(skip)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    bias_d = ke.DeviceArray(bias)
    y_d = ke.DeviceArray(output_y)
    optional_d = ke.DeviceArray(output_optional)
    f = getattr(ke, func)

    my_op = f(
        y_d,
        optional_d,
        input_d,
        skip_d,
        gamma_d,
        beta_d,
        bias_d,
        epsilon,
        hidden_size,
        batch_size * seq_len * hidden_size,
    )

    duration_ms = -1
    if my_op.IsSupported():
        duration_ms = my_op.Profile()
    total_bytes = (input_x.size * 3 + bias.size * 3) * dtype_to_bytes(dtype)

    ke.report(SkipLayerNormMetric(func, dtype, duration_ms, total_bytes, batch_size, seq_len, hidden_size))


def profile_with_args(batch_size, seq_len, hidden_size, dtype, has_optional_output=False, simplified=False):
    with ke.benchmark():
        for func in dtype_to_funcs(dtype, simplified):
            profile_skip_layer_norm_func(batch_size, seq_len, hidden_size, dtype, func, has_optional_output)


def profile():
    for dtype in dtypes:
        for bert_size in get_bert_sizes_profile():
            profile_with_args(*bert_size, dtype)
            print()


if __name__ == "__main__":
    parser = ke.get_argument_parser()
    group = parser.add_argument_group()
    group.add_argument("batch_size", type=int)
    group.add_argument("seq_len", type=int)
    group.add_argument("hidden_size", type=int)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--has_optional_output", "-o", action="store_true")
    group.add_argument("--simplified", "-s", action="store_true", default=False)

    if not ke.has_args():
        profile()
    else:
        args = parser.parse_args()
        args.func(
            args.batch_size,
            args.seq_len,
            args.hidden_size,
            args.dtype,
            args.has_optional_output,
            args.simplified,
        )
