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


def get_test_sizes():
    batch_count = [1, 8]
    softmax_elements = [1, 2, 3, 4, 5, 7, 8, 9, 11, 16, 31, 32, 33, 64, 65, 127, 128, 1024, 1025, 2048, 4096]
    is_log_softmax = [True, False]
    return product(batch_count, softmax_elements, is_log_softmax)


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: re.match("Softmax.*_half.*", x), dir(ke))),
        "float32": list(filter(lambda x: re.match("Softmax.*_float.*", x), dir(ke))),
    }
    return type_map[dtype]


def softmax(x, is_log_softmax):
    x = x - np.max(x, axis=-1, keepdims=1)
    if is_log_softmax:
        return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=1))
    return (np.exp(x)) / np.sum(np.exp(x), axis=-1, keepdims=1)


def run_softmax(batch_count, softmax_elements, is_log_softmax, dtype, func):
    np.random.seed(0)
    x = np.random.rand(batch_count, softmax_elements).astype(dtype)
    y = np.random.rand(batch_count, softmax_elements).astype(dtype)

    x_d = ke.DeviceArray(x)
    y_d = ke.DeviceArray(y)

    softmax_func = getattr(ke, func)
    softmax_op = softmax_func(
        y_d, x_d, softmax_elements, softmax_elements, softmax_elements, batch_count, is_log_softmax
    )
    if softmax_op.IsSupported():
        softmax_op.Run()
        y_d.UpdateHostNumpyArray()

        y_ref = softmax(x, is_log_softmax)
        np.testing.assert_allclose(y_ref, y, rtol=1e-02)


dtypes = ["float16", "float32"]


@pytest.mark.parametrize("batch_count, softmax_elements, is_log_softmax", get_test_sizes())
@pytest.mark.parametrize("dtype", dtypes)
def test_softmax(batch_count, softmax_elements, is_log_softmax, dtype):
    for f in dtype_to_funcs(dtype):
        run_softmax(batch_count, softmax_elements, is_log_softmax, dtype, f)


@dataclass
class SoftmaxMetric(ke.BandwidthMetric):
    batch_count: int
    softmax_elements: int
    is_log_softmax: bool

    def report(self):
        prefix = f"{self.name:<50} {self.dtype} batch_count={self.batch_count:<4} softmax_elements={self.softmax_elements:<4} "
        if self.duration > 0:
            return prefix + f"{self.duration:.2f} us, {self.gbps:.2f} GB/s"
        return prefix + "not supported or redundant"


def profile_softmax_func(batch_count, softmax_elements, is_log_softmax, dtype, func):
    np.random.seed(0)
    x = np.random.rand(batch_count, softmax_elements).astype(dtype)
    y = np.random.rand(batch_count, softmax_elements).astype(dtype)

    x_d = ke.DeviceArray(x)
    y_d = ke.DeviceArray(y)

    softmax_func = getattr(ke, func)
    softmax_op = softmax_func(
        y_d, x_d, softmax_elements, softmax_elements, softmax_elements, batch_count, is_log_softmax
    )
    if softmax_op.IsSupported():
        duration_ms = softmax_op.Profile()
    total_bytes = 2 * batch_count * softmax_elements * dtype_to_bytes(dtype)

    ke.report(SoftmaxMetric(func, dtype, duration_ms, total_bytes, batch_count, softmax_elements, is_log_softmax))


def profile_with_args(batch_count, softmax_elements, is_log_softmax, dtype, sort):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(dtype):
            profile_softmax_func(batch_count, softmax_elements, is_log_softmax, dtype, func)


profile_size = [(1, 2048), (8, 2048), (65536, 4096)]


def profile():
    for dtype in dtypes:
        for batch_count, softmax_elements in profile_size:
            profile_with_args(batch_count, softmax_elements, True, dtype, True)
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("batch_count", type=int)
    group.add_argument("softmax_elements", type=int)
    group.add_argument("is_log_softmax", type=int)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.batch_count, args.softmax_elements, args.is_log_softmax, args.dtype, args.sort)
