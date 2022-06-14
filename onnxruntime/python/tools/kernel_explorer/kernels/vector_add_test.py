# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys

sys.path.append("../build")

import kernel_explorer as ke
import numpy as np
import pytest


def dtype_to_bytes(dtype):
    type_map = {
        "float16": 2,
        "float32": 4,
    }
    return type_map[dtype]


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: "VectorAdd_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "VectorAdd_float" in x, dir(ke))),
    }
    return type_map[dtype]


@pytest.mark.skip(reason="called by test_vector_add_all_sizes")
def test_vector_add(size, dtype, func):
    np.random.seed(0)
    x = np.random.rand(size).astype(dtype)
    y = np.random.rand(size).astype(dtype)
    z = np.random.rand(size).astype(dtype)

    x_d = ke.DeviceArray(x)
    y_d = ke.DeviceArray(y)
    z_d = ke.DeviceArray(z)
    f = getattr(ke, func)
    va = f(x_d, y_d, z_d, size)
    va.Run()
    z_d.UpdateHostNumpyArray()

    z_ref = x + y
    np.testing.assert_allclose(z_ref, z)


@pytest.mark.parametrize("size", [1, 3, 4, 16, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1024])
def test_vector_add_all_sizes(size):
    dtypes = ["float16", "float32"]
    for dtype in dtypes:
        for f in dtype_to_funcs(dtype):
            test_vector_add(size, dtype, f)


def profile_vector_add_func(size, dtype, func):
    np.random.seed(0)
    x = np.random.rand(size).astype(dtype)
    y = np.random.rand(size).astype(dtype)
    z = np.random.rand(size).astype(dtype)

    x_d = ke.DeviceArray(x)
    y_d = ke.DeviceArray(y)
    z_d = ke.DeviceArray(z)
    f = getattr(ke, func)
    va = f(x_d, y_d, z_d, size)
    t = va.Profile()
    print(dtype, size, f, f"{t*1000:.2f} us", f"{size*3*(dtype_to_bytes(dtype))*1e3/t/1e9:.2f} GB/s")


def profile():
    sizes = [10000, 100000, 1000000, 10000000]
    dtypes = ["float16", "float32"]
    for dt in dtypes:
        for s in sizes:
            for f in dtype_to_funcs(dt):
                profile_vector_add_func(s, dt, f)
            print()
        print()


if __name__ == "__main__":
    profile()
