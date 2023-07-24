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
from utils import dtype_to_bytes, dtype_to_suffix, standardization


def get_sd_sizes():
    batch_sizes = [1, 2]
    height = [8, 16, 32]
    num_channels = [320, 640, 1280, 1920, 2560]

    num_groups = [32]
    return product(batch_sizes, height, num_channels, num_groups)


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: re.match("GroupNormNHWC.*_half", x), dir(ke))),
        "float32": list(filter(lambda x: re.match("GroupNormNHWC.*_float", x), dir(ke))),
    }
    return type_map[dtype]


def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))


def group_norm(input_x, gamma, beta, num_groups, epsilon, with_swish):
    n, h, w, c = input_x.shape
    input_x = input_x.transpose([0, 3, 1, 2])
    assert c % num_groups == 0
    x = input_x.reshape((n, num_groups, -1))
    x = standardization(x, -1, epsilon)
    x = x.reshape((n, c, h, w))
    x = x.transpose([0, 2, 3, 1])
    x = x * gamma + beta

    if with_swish:
        x = x * sigmoid_function(x)
    return x


def run_group_norm(batch_size: int, height: int, num_channels: int, num_groups: int, dtype: str, swish: bool, func):
    np.random.seed(0)
    width = height
    input_x = np.random.rand(batch_size, height, width, num_channels).astype(np.float32)
    gamma = np.random.rand(num_channels).astype(np.float32)
    beta = np.random.rand(num_channels).astype(np.float32)
    # the size of workspace is defined in onnxruntime/contrib_ops/cuda/diffusion/group_norm_impl.h L18
    workspace = np.random.rand((np.dtype(np.float32).itemsize * 2) * 32 * 32).astype(np.float32)
    epsilon = 1e-05
    output_y = np.random.rand(batch_size, height, width, num_channels).astype(dtype)
    use_swish = swish

    host_x = input_x.astype(dtype)
    input_d = ke.DeviceArray(host_x)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    workspace_d = ke.DeviceArray(workspace)
    y_d = ke.DeviceArray(output_y)
    f = getattr(ke, func)

    my_op = f(
        y_d,
        workspace_d,
        input_d,
        gamma_d,
        beta_d,
        batch_size,
        height,
        width,
        num_channels,
        num_groups,
        epsilon,
        use_swish,
    )
    y_ref = group_norm(input_x, gamma, beta, num_groups, epsilon, use_swish).astype(dtype)

    for impl in my_op.ListOps():
        if not my_op.SelectOp(impl):
            continue

        my_op.Run()

        y_d.UpdateHostNumpyArray()

        np.testing.assert_allclose(y_ref, output_y, atol=1e-02)


dtypes = ["float32", "float16"]


@pytest.mark.parametrize("sd_sizes", get_sd_sizes())
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("swish", [True])
def test_group_norm(sd_sizes, dtype, swish):
    for func in dtype_to_funcs(dtype):
        run_group_norm(*sd_sizes, dtype, swish, func)


@pytest.mark.parametrize("sd_sizes", get_sd_sizes())
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("swish", [True])
def test_group_norm_ck(sd_sizes, dtype, swish):
    swish_suffix = "Swish" if swish else "Pass"
    ck_f_name = "CKGroupNormNHWC" + swish_suffix + "_" + dtype_to_suffix(dtype)
    run_group_norm(*sd_sizes, dtype, swish, ck_f_name)


@dataclass
class GroupNormNHWCMetric(ke.BandwidthMetric):
    batch_size: int
    height: int
    width: int
    num_channels: int
    groups: int

    def report(self):
        common = (
            f"{self.dtype:<4} batch={self.batch_size:<4} height={self.height:<4} width={self.width:<4}"
            f"num_channels={self.num_channels:<6} groups={self.groups:<4} {self.name}"
        )
        if self.duration > 0:
            return f"{self.duration:.2f} us, {self.gbps:.2f} GB/s  " + common
        return "not supported          " + common


def profile_group_norm_func(
    batch_size: int, height: int, width: int, num_channels: int, num_groups: int, dtype: str, swish: bool, func
):
    np.random.seed(0)
    input_x = np.random.rand(batch_size, height, width, num_channels).astype(dtype)
    gamma = np.random.rand(num_channels).astype(np.float32)
    beta = np.random.rand(num_channels).astype(np.float32)
    workspace = np.random.rand(np.dtype(np.float32).itemsize * 2 * 32 * 32).astype(np.float32)
    epsilon = 0.05
    output_y = np.random.rand(batch_size, height, width, num_channels).astype(dtype)
    use_swish = swish

    input_d = ke.DeviceArray(input_x)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    workspace_d = ke.DeviceArray(workspace)
    y_d = ke.DeviceArray(output_y)
    f = getattr(ke, func)

    my_op = f(
        y_d,
        workspace_d,
        input_d,
        gamma_d,
        beta_d,
        batch_size,
        height,
        width,
        num_channels,
        num_groups,
        epsilon,
        use_swish,
    )
    for impl in my_op.ListOps():
        duration_ms = -1
        if my_op.SelectOp(impl):
            duration_ms = my_op.Profile()
        total_bytes = (input_x.size * 2 + gamma.size * 2) * dtype_to_bytes(dtype)

        ke.report(
            GroupNormNHWCMetric(
                impl, dtype, duration_ms, total_bytes, batch_size, height, width, num_channels, num_groups
            )
        )


def profile_with_args(batch_size, height, width, num_channels, num_groups, dtype, swish=True, sort=True):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(dtype):
            profile_group_norm_func(batch_size, height, width, num_channels, num_groups, dtype, swish, func)
        # ck function
        swish_suffix = "Swish" if swish else "Pass"
        ck_f_name = "CKGroupNormNHWC" + swish_suffix + "_" + dtype_to_suffix(dtype)
        profile_group_norm_func(batch_size, height, width, num_channels, num_groups, dtype, swish, ck_f_name)


sd_profile_sizes = [
    (2, 64, 64, 320, 32),
    (2, 32, 32, 640, 32),
    (2, 16, 16, 1280, 32),
    (2, 64, 64, 640, 32),
    (2, 16, 16, 2560, 32),
    (2, 32, 32, 1280, 32),
    (2, 32, 32, 1920, 32),
    (2, 8, 8, 1280, 32),
    (2, 64, 64, 960, 32),
    (2, 32, 32, 960, 32),
    (2, 32, 32, 320, 32),
    (2, 16, 16, 640, 32),
    (2, 16, 16, 1920, 32),
    (2, 8, 8, 2560, 32),
]


def profile():
    for dtype in dtypes:
        for sd_size in sd_profile_sizes:
            profile_with_args(*sd_size, dtype)
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("batch_size", type=int)
    group.add_argument("height", type=int)
    group.add_argument("width", type=int)
    group.add_argument("num_channels", type=int)
    group.add_argument("num_groups", type=int)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--swish", action="store_true")
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(
            args.batch_size,
            args.height,
            args.width,
            args.num_channels,
            args.num_groups,
            args.dtype,
            args.swish,
            args.sort,
        )
