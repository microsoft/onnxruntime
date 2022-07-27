# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import kernel_explorer as ke
import numpy as np
import pytest


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: "FastGelu_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "FastGelu_float" in x, dir(ke))),
        "float64": list(filter(lambda x: "FastGelu_double" in x, dir(ke))),
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
    va = f(x_d, bias_d, y_d, x.size, bias.size)
    va.Run()
    y_d.UpdateHostNumpyArray()

    y_ref = fast_gelu(x, bias)
    np.testing.assert_allclose(y_ref, y, rtol=1e-02)


test_cases = [((2, 16), 16), ((1, 2, 768), 768), ((1, 2, 1024), 1024)]


@pytest.mark.parametrize("x_size, bias_size", test_cases)
def test_fast_gelu(x_size, bias_size):
    dtypes = ["float16", "float32", "float64"]
    for dtype in dtypes:
        for f in dtype_to_funcs(dtype):
            run_fast_gelu(x_size, bias_size, dtype, f)


def profile_vector_add_func(batch_size, seq_len, hidden_size, dtype, func):
    x_size = [batch_size, seq_len, hidden_size * 3]
    bias_size = hidden_size * 3
    np.random.seed(0)
    x = np.random.rand(*x_size).astype(dtype)
    bias = np.random.rand(bias_size).astype(dtype)
    y = np.random.rand(*x_size).astype(dtype)

    x_d = ke.DeviceArray(x)
    bias_d = ke.DeviceArray(bias)
    y_d = ke.DeviceArray(y)
    f = getattr(ke, func)
    va = f(x_d, bias_d, y_d, x.size, bias.size)
    t = va.Profile()
    print(
        dtype,
        batch_size,
        seq_len,
        hidden_size,
        f,
        f"{t*1000:.2f} us",
        f"{(x.size*2+bias.size)*x.itemsize*1e3/t/1e9:.2f} GB/s",
    )


def profile():
    batch_size = [1]
    seq_len = [384]
    hidden_size = [1024]
    dtypes = ["float16", "float32", "float64"]
    for dt in dtypes:
        for bs in batch_size:
            for sl in seq_len:
                for hs in hidden_size:
                    for f in dtype_to_funcs(dt):
                        profile_vector_add_func(bs, sl, hs, dt, f)
                    print()


if __name__ == "__main__":
    profile()
