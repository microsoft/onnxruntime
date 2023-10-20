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
from utils import dtype_to_bytes, dtype_to_suffix, softmax


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


def _test_softmax(batch_count, softmax_elements, is_log_softmax, dtype, func):
    np.random.seed(0)
    x = np.random.rand(batch_count, softmax_elements).astype(dtype)
    y = np.random.rand(batch_count, softmax_elements).astype(dtype)

    x_d = ke.DeviceArray(x)
    y_d = ke.DeviceArray(y)
    y_ref = softmax(x, is_log_softmax=is_log_softmax)

    softmax_func = getattr(ke, func)
    softmax_op = softmax_func(
        y_d, x_d, softmax_elements, softmax_elements, softmax_elements, batch_count, is_log_softmax
    )
    for impl in softmax_op.ListOps():
        if not softmax_op.SelectOp(impl):
            continue

        softmax_op.Run()
        y_d.UpdateHostNumpyArray()

        np.testing.assert_allclose(y_ref, y, rtol=1e-02, err_msg=func)


dtypes = ["float16", "float32"]


@pytest.mark.parametrize("batch_count, softmax_elements, is_log_softmax", get_test_sizes())
@pytest.mark.parametrize("dtype", dtypes)
def test_softmax(batch_count, softmax_elements, is_log_softmax, dtype):
    for f in dtype_to_funcs(dtype):
        _test_softmax(batch_count, softmax_elements, is_log_softmax, dtype, f)


@pytest.mark.parametrize("batch_count, softmax_elements, is_log_softmax", get_test_sizes())
@pytest.mark.parametrize("dtype", dtypes)
def test_ck_softmax(batch_count, softmax_elements, is_log_softmax, dtype):
    ck_f_name = "CKSoftmax_" + dtype_to_suffix(dtype)
    _test_softmax(batch_count, softmax_elements, is_log_softmax, dtype, ck_f_name)


@dataclass
class SoftmaxMetric(ke.BandwidthMetric):
    batch_count: int
    softmax_elements: int
    is_log_softmax: bool

    def report(self):
        common = f"{self.dtype} batch_count={self.batch_count:<4} softmax_elements={self.softmax_elements:<4} is_log_softmax={self.is_log_softmax:<4} {self.name}"
        if self.duration > 0:
            return f"{self.duration:6.2f} us {self.gbps:5.2f} GB/s " + common
        return "not supported        " + common


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

    for impl in softmax_op.ListOps():
        duration_ms = -1
        if softmax_op.SelectOp(impl):
            duration_ms = softmax_op.Profile()
        total_bytes = 2 * batch_count * softmax_elements * dtype_to_bytes(dtype)

        ke.report(SoftmaxMetric(impl, dtype, duration_ms, total_bytes, batch_count, softmax_elements, is_log_softmax))


def profile_with_args(batch_count, softmax_elements, is_log_softmax, dtype):
    with ke.benchmark():
        for func in dtype_to_funcs(dtype):
            profile_softmax_func(batch_count, softmax_elements, is_log_softmax, dtype, func)
        # ck function
        ck_f_name = "CKSoftmax_" + dtype_to_suffix(dtype)
        profile_softmax_func(batch_count, softmax_elements, is_log_softmax, dtype, ck_f_name)


profile_size = [(1, 2048), (8, 2048), (65536, 4096)]


def profile():
    for dtype in dtypes:
        for batch_count, softmax_elements in profile_size:
            profile_with_args(batch_count, softmax_elements, False, dtype)
            print()


if __name__ == "__main__":
    parser = ke.get_argument_parser()
    group = parser.add_argument_group()
    group.add_argument("batch_count", type=int)
    group.add_argument("softmax_elements", type=int)
    group.add_argument("is_log_softmax", type=int)
    group.add_argument("dtype", choices=dtypes)

    if not ke.has_args():
        profile()
    else:
        args = parser.parse_args()
        args.func(args.batch_count, args.softmax_elements, args.is_log_softmax, args.dtype)
