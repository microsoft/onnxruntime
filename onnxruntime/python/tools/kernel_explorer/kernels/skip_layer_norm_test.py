# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import kernel_explorer as ke
import numpy as np
import pytest


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: "SkipLayerNorm_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "SkipLayerNorm_float" in x, dir(ke))),
    }
    return type_map[dtype]


def skip_layer_norm(input, skip, bias, gamma, beta, epsilon):
    val = input + skip + bias
    u = np.mean(val, axis=(2,))
    s = np.var(val, axis=(2,))
    y = val - u[..., None]
    y = y / np.sqrt(s + epsilon)[..., None]
    y = y * gamma + beta
    return y


def run_skip_layer_norm(batch_size, seq_len, hidden_size, dtype, func):
    np.random.seed(0)
    x = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    skip = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    bias = np.random.rand(hidden_size).astype(dtype)
    gamma = np.random.rand(hidden_size).astype(dtype)
    beta = np.random.rand((hidden_size)).astype(dtype)
    epsilon = 0.0005
    y = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)

    input_d = ke.DeviceArray(x)
    skip_d = ke.DeviceArray(skip)
    bias_d = ke.DeviceArray(bias)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    y_d = ke.DeviceArray(y)
    f = getattr(ke, func)
    va = f(y_d, input_d, skip_d, gamma_d, beta_d, bias_d, epsilon, hidden_size, batch_size * seq_len * hidden_size)
    va.Run()
    y_d.UpdateHostNumpyArray()

    y_ref = skip_layer_norm(x, skip, bias, gamma, beta, epsilon)
    np.testing.assert_almost_equal(y_ref, y, decimal=1e-05)


batch_size = [1, 8, 16, 32, 64, 128]
seq_len = [256, 384]
hidden_size = [256, 384, 1024]


@pytest.mark.parametrize("batch_size", batch_size)
@pytest.mark.parametrize("seq_len", seq_len)
@pytest.mark.parametrize("hidden_size", hidden_size)
def test_skip_layer_norm(batch_size, seq_len, hidden_size):
    dtypes = ["float32", "float16"]
    for dtype in dtypes:
        for f in dtype_to_funcs(dtype):
            run_skip_layer_norm(batch_size, seq_len, hidden_size, dtype, f)


def profile_skip_layer_norm_func(batch_size, seq_len, hidden_size, dtype, func):
    np.random.seed(0)
    x = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    skip = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)
    bias = np.random.rand(hidden_size).astype(dtype)
    gamma = np.random.rand(hidden_size).astype(dtype)
    beta = np.random.rand(hidden_size).astype(dtype)
    bias = np.random.rand(hidden_size).astype(dtype)
    epsilon = 0.0005
    y = np.random.rand(batch_size, seq_len, hidden_size).astype(dtype)

    input_d = ke.DeviceArray(x)
    skip_d = ke.DeviceArray(skip)
    bias_d = ke.DeviceArray(bias)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    bias_d = ke.DeviceArray(bias)
    y_d = ke.DeviceArray(y)
    f = getattr(ke, func)
    va = f(y_d, input_d, skip_d, gamma_d, beta_d, bias_d, epsilon, hidden_size, batch_size * seq_len * hidden_size)
    t = va.Profile()
    print(
        dtype,
        batch_size,
        seq_len,
        hidden_size,
        f,
        f"{t * 1000:.2f} us",
        f"{(x.size * 3 + bias.size * 3) * x.itemsize * 1e3 / t / 1e9:.2f} GB/s",
    )


def profile():
    batch_size = [1]
    seq_len = [384]
    hidden_size = [1024]
    dtypes = ["float32"]
    for dt in dtypes:
        for bs in batch_size:
            for sl in seq_len:
                for hs in hidden_size:
                    for f in dtype_to_funcs(dt):
                        profile_skip_layer_norm_func(bs, sl, hs, dt, f)
                    print()


if __name__ == "__main__":
    profile()
