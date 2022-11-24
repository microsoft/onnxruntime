# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys

import kernel_explorer as ke
import numpy as np
import pytest
from utils import sort_profile_results


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


def run_vector_add(size, dtype, func):
    np.random.seed(0)
    x = np.random.rand(size).astype(dtype)
    y = np.random.rand(size).astype(dtype)
    z = np.random.rand(size).astype(dtype)

    x_d = ke.DeviceArray(x)
    y_d = ke.DeviceArray(y)
    z_d = ke.DeviceArray(z)
    f = getattr(ke, func)
    my_op = f(x_d, y_d, z_d, size)
    my_op.Run()
    z_d.UpdateHostNumpyArray()

    z_ref = x + y
    np.testing.assert_allclose(z_ref, z)


dtypes = ["float16", "float32"]


@pytest.mark.parametrize("size", [1, 3, 4, 16, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1024])
@pytest.mark.parametrize("dtype", dtypes)
def test_vector_add(size, dtype):
    for dtype in dtypes:
        for f in dtype_to_funcs(dtype):
            run_vector_add(size, dtype, f)


def profile_vector_add_func(size, dtype, func):
    np.random.seed(0)
    x = np.random.rand(size).astype(dtype)
    y = np.random.rand(size).astype(dtype)
    z = np.random.rand(size).astype(dtype)

    x_d = ke.DeviceArray(x)
    y_d = ke.DeviceArray(y)
    z_d = ke.DeviceArray(z)
    f = getattr(ke, func)
    my_op = f(x_d, y_d, z_d, size)
    duration = my_op.Profile()
    gbytes_per_seconds = size * 3 * (dtype_to_bytes(dtype)) * 1e3 / duration / 1e9
    duration = duration * 1000
    return {"func": func, "duration": duration, "GBps": gbytes_per_seconds}


def print_results(size, dtype, profile_results):
    for result in profile_results:
        print(
            f"{result['func']:<50} {dtype} size={size:<4}",
            f"{result['duration']:.2f} us",
            f"{result['GBps']:.2f} GB/s",
        )


def profile_with_args(size, dtype, enable_sort=True):
    if enable_sort:
        profile_results = []
        for func in dtype_to_funcs(dtype):
            profile_result = profile_vector_add_func(size, dtype, func)
            profile_results.append(profile_result)
        sorted_profile_results = sort_profile_results(profile_results, sort_item="GBps", reverse=True)
        print_results(size, dtype, sorted_profile_results)
    else:
        for func in dtype_to_funcs(dtype):
            profile_result = profile_vector_add_func(size, dtype, func)
            print_results(size, dtype, [profile_result])
    print()


def profile():
    sizes = [10000, 100000, 1000000, 10000000]
    for dt in dtypes:
        for s in sizes:
            profile_with_args(s, dt)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("size", type=int)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--enable_sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.size, args.dtype, args.enable_sort)
