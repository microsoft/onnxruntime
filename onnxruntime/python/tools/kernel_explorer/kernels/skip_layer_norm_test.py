# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re
from itertools import product

import kernel_explorer as ke
import numpy as np
import pytest


def get_bert_sizes():
    batch_sizes = [1, 8, 64, 128]
    seq_lens = [64, 128, 256, 384, 512]
    hidden_sizes = [768, 1024]
    return product(batch_sizes, seq_lens, hidden_sizes)


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: re.search("SkipLayerNorm.*_half", x), dir(ke))),
        "float32": list(filter(lambda x: re.search("SkipLayerNorm.*_float", x), dir(ke))),
    }
    return type_map[dtype]


def skip_layer_norm(input_x, skip, bias, gamma, beta, epsilon):
    val = input_x + skip + bias
    x_u = np.mean(val, axis=(2,))
    x_s = np.var(val, axis=(2,))
    output = val - x_u[..., None]
    output = output / np.sqrt(x_s + epsilon)[..., None]
    output = output * gamma + beta
    return output


def run_skip_layer_norm(batch_size: int, seq_len: int, hidden_size: int, dtype: str, func):
    np.random.seed(0)
    input_x = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    skip = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    bias = np.random.rand(hidden_size).astype(dtype)
    gamma = np.random.rand(hidden_size).astype(dtype)
    beta = np.random.rand((hidden_size)).astype(dtype)
    epsilon = 0.0005
    output_y = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)

    input_d = ke.DeviceArray(input_x)
    skip_d = ke.DeviceArray(skip)
    bias_d = ke.DeviceArray(bias)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    y_d = ke.DeviceArray(output_y)
    my_func = getattr(ke, func)
    my_op = my_func(
        y_d, input_d, skip_d, gamma_d, beta_d, bias_d, epsilon, hidden_size, batch_size * seq_len * hidden_size
    )
    if my_op.IsSupported():
        my_op.Run()

        y_d.UpdateHostNumpyArray()

        y_ref = skip_layer_norm(input_x, skip, bias, gamma, beta, epsilon)
        np.testing.assert_almost_equal(y_ref, output_y, decimal=1e-05)


dtypes = ["float32", "float16"]


@pytest.mark.parametrize("bert_sizes", get_bert_sizes())
@pytest.mark.parametrize("dtype", dtypes)
def test_skip_layer_norm(bert_sizes, dtype):
    for func in dtype_to_funcs(dtype):
        print(func)
        run_skip_layer_norm(*bert_sizes, dtype, func)


def profile_skip_layer_norm_func(batch_size, seq_len, hidden_size, dtype, func):
    np.random.seed(0)
    input_x = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    skip = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    gamma = np.random.rand(hidden_size).astype(dtype)
    beta = np.random.rand(hidden_size).astype(dtype)
    bias = np.random.rand(hidden_size).astype(dtype)
    epsilon = 0.0005
    output_y = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)

    input_d = ke.DeviceArray(input_x)
    skip_d = ke.DeviceArray(skip)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    bias_d = ke.DeviceArray(bias)
    y_d = ke.DeviceArray(output_y)
    my_func = getattr(ke, func)
    my_op = my_func(
        y_d, input_d, skip_d, gamma_d, beta_d, bias_d, epsilon, hidden_size, batch_size * seq_len * hidden_size
    )
    if my_op.IsSupported():
        duration = my_op.Profile()
        print(
            dtype,
            batch_size,
            seq_len,
            hidden_size,
            my_func,
            f"{duration * 1000:.2f} us",
            f"{(input_x.size * 3 + bias.size * 3) * input_x.itemsize * 1e3 / duration / 1e9:.2f} GB/s",
        )


def profile():
    for dtype in dtypes:
        for bert_size in get_bert_sizes():
            for func in dtype_to_funcs(dtype):
                profile_skip_layer_norm_func(*bert_size, dtype, func)
            print()


if __name__ == "__main__":
    profile()
