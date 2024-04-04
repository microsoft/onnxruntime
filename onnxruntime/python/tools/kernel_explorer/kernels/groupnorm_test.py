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


def group_norm(input_x, skip_x, bias_x, gamma, beta, num_groups, epsilon, with_silu, has_skip):
    add_output = None
    if has_skip:
        input_x = input_x + skip_x + bias_x
        add_output = input_x
    n, h, w, c = input_x.shape
    input_x = input_x.transpose([0, 3, 1, 2])
    assert c % num_groups == 0
    x = input_x.reshape((n, num_groups, -1))
    x = standardization(x, -1, epsilon)
    x = x.reshape((n, c, h, w))
    x = x.transpose([0, 2, 3, 1])
    x = x * gamma + beta

    if with_silu:
        x = x * sigmoid_function(x)
    return x, add_output


def run_group_norm(
    batch_size: int, height: int, num_channels: int, num_groups: int, dtype: str, silu: bool, has_skip: bool, func
):
    np.random.seed(0)
    width = height
    input_x = np.random.rand(batch_size, height, width, num_channels).astype(np.float32)
    gamma = np.random.rand(num_channels).astype(np.float32)
    beta = np.random.rand(num_channels).astype(np.float32)
    # the size of workspace is defined in onnxruntime/contrib_ops/cuda/diffusion/group_norm_impl.h L18
    workspace = np.random.rand((np.dtype(np.float32).itemsize * 2) * batch_size * num_groups).astype(np.float32)
    epsilon = 1e-05
    output_y = np.random.rand(batch_size, height, width, num_channels).astype(dtype)

    skip_x = (
        np.random.rand(batch_size, height, width, num_channels).astype(np.float32)
        if has_skip
        else np.empty((0), dtype=dtype)
    )
    bias_x = np.random.rand(num_channels).astype(np.float32) if has_skip else np.empty((0), dtype=dtype)
    add_output = (
        np.random.rand(batch_size, height, width, num_channels).astype(dtype)
        if has_skip
        else np.empty((0), dtype=dtype)
    )
    use_silu = silu
    broadcast_skip = False
    if has_skip:
        skip_x_shape = skip_x.shape
        b2 = len(skip_x_shape) == 2 and skip_x_shape[0] == batch_size and skip_x_shape[1] == num_channels
        b4 = (
            len(skip_x_shape) == 4
            and skip_x_shape[0] == batch_size
            and skip_x_shape[1] == 1
            and skip_x_shape[2] == 1
            and skip_x_shape[3] == num_channels
        )
        if b2 or b4:
            broadcast_skip = True
    channels_per_block = 0  # Compute in params initialization

    input_d = ke.DeviceArray(input_x.astype(dtype))
    skip_d = ke.DeviceArray(skip_x.astype(dtype))
    bias_d = ke.DeviceArray(bias_x.astype(dtype))
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    workspace_d = ke.DeviceArray(workspace)
    y_d = ke.DeviceArray(output_y)
    y_add_d = ke.DeviceArray(add_output)
    f = getattr(ke, func)

    my_op = f(
        y_d,
        y_add_d,
        input_d,
        skip_d,
        bias_d,
        gamma_d,
        beta_d,
        workspace_d,
        epsilon,
        batch_size,
        num_channels,
        height,
        width,
        num_groups,
        use_silu,
        broadcast_skip,
        channels_per_block,
    )
    y_ref, y_add_d_ref = group_norm(input_x, skip_x, bias_x, gamma, beta, num_groups, epsilon, use_silu, has_skip)
    y_ref = y_ref.astype(dtype)

    for impl in my_op.ListOps():
        if not my_op.SelectOp(impl):
            continue

        my_op.Run()

        y_d.UpdateHostNumpyArray()

        np.testing.assert_allclose(y_ref, output_y, atol=1e-02)
        if has_skip:
            y_add_d_ref = y_add_d_ref.astype(dtype)
            y_add_d.UpdateHostNumpyArray()
            np.testing.assert_allclose(y_add_d_ref, add_output, atol=1e-02)


dtypes = ["float32", "float16"]


@pytest.mark.parametrize("sd_sizes", get_sd_sizes())
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("silu", [True])
@pytest.mark.parametrize("has_skip", [True, False])
def test_group_norm(sd_sizes, dtype, silu, has_skip):
    for func in dtype_to_funcs(dtype):
        run_group_norm(*sd_sizes, dtype, silu, has_skip, func)


@pytest.mark.parametrize("sd_sizes", get_sd_sizes())
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("silu", [True])
@pytest.mark.parametrize("has_skip", [False])
def test_group_norm_ck(sd_sizes, dtype, silu, has_skip):
    silu_suffix = "Silu" if silu else "Pass"
    ck_f_name = "CKGroupNormNHWC" + silu_suffix + "_" + dtype_to_suffix(dtype)
    run_group_norm(*sd_sizes, dtype, silu, has_skip, ck_f_name)


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
            return f"{self.duration:>6.2f} us, {self.gbps:>5.2f} GB/s  " + common
        return "not supported          " + common


def profile_group_norm_func(
    batch_size: int,
    height: int,
    width: int,
    num_channels: int,
    num_groups: int,
    dtype: str,
    silu: bool,
    has_skip: bool,
    func,
):
    np.random.seed(0)
    input_x = np.random.rand(batch_size, height, width, num_channels).astype(dtype)
    gamma = np.random.rand(num_channels).astype(np.float32)
    beta = np.random.rand(num_channels).astype(np.float32)
    workspace = np.random.rand(np.dtype(np.float32).itemsize * 2 * batch_size * num_groups).astype(np.float32)
    epsilon = 0.05
    output_y = np.random.rand(batch_size, height, width, num_channels).astype(dtype)

    skip_x = (
        np.random.rand(batch_size, height, width, num_channels).astype(dtype)
        if has_skip
        else np.empty((0), dtype=dtype)
    )
    bias_x = np.random.rand(num_channels).astype(dtype) if has_skip else np.empty((0), dtype=dtype)
    add_output = (
        np.random.rand(batch_size, height, width, num_channels).astype(dtype)
        if has_skip
        else np.empty((0), dtype=dtype)
    )
    use_silu = silu
    broadcast_skip = False
    channels_per_block = 0  # Compute in params initialization

    input_d = ke.DeviceArray(input_x)
    skip_d = ke.DeviceArray(skip_x)
    bias_d = ke.DeviceArray(bias_x)
    gamma_d = ke.DeviceArray(gamma)
    beta_d = ke.DeviceArray(beta)
    workspace_d = ke.DeviceArray(workspace)
    y_d = ke.DeviceArray(output_y)
    y_add_d = ke.DeviceArray(add_output)
    f = getattr(ke, func)

    my_op = f(
        y_d,
        y_add_d,
        input_d,
        skip_d,
        bias_d,
        gamma_d,
        beta_d,
        workspace_d,
        epsilon,
        batch_size,
        num_channels,
        height,
        width,
        num_groups,
        use_silu,
        broadcast_skip,
        channels_per_block,
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


def profile_with_args(batch_size, height, width, num_channels, num_groups, dtype, silu=True, has_skip=True, sort=True):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(dtype):
            profile_group_norm_func(batch_size, height, width, num_channels, num_groups, dtype, silu, has_skip, func)
        # ck function
        silu_suffix = "Silu" if silu else "Pass"
        ck_f_name = "CKGroupNormNHWC" + silu_suffix + "_" + dtype_to_suffix(dtype)
        profile_group_norm_func(batch_size, height, width, num_channels, num_groups, dtype, silu, has_skip, ck_f_name)


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
    group.add_argument("--silu", action="store_true")
    group.add_argument("--has_skip", action="store_true")
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
            args.silu,
            args.has_skip,
            args.sort,
        )
