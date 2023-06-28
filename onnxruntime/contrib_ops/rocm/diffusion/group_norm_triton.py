# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from itertools import product

import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    input_ptr,
    output_ptr,
    gamma_ptr,
    beta_ptr,
    img_size,
    c,
    c_per_group,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HW_SIZE: tl.constexpr,
    ACTIVATION_SWISH: tl.constexpr,
):
    row_x = tl.program_id(0)
    row_y = tl.program_id(1)
    stride = img_size * c
    input_ptr += row_x * stride + row_y * c_per_group
    output_ptr += row_x * stride + row_y * c_per_group
    gamma_ptr += row_y * c_per_group
    beta_ptr += row_y * c_per_group

    cols = tl.arange(0, BLOCK_SIZE)
    hw = tl.arange(0, HW_SIZE)
    offsets = hw[:, None] * c + cols[None, :]
    mask = (cols < c_per_group)[None, :]

    # Calculate mean and variance
    _sum = tl.zeros([HW_SIZE, BLOCK_SIZE], dtype=tl.float32)
    _square_sum = tl.zeros([HW_SIZE, BLOCK_SIZE], dtype=tl.float32)
    for i in range(tl.cdiv(img_size, HW_SIZE)):
        x_ptr = input_ptr + i * HW_SIZE * c
        a = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        _sum += a
        _square_sum += a * a

    # Set axis=None (or leave it unspecified) to reduce all axes.
    # TODO: In older Triton we have to reduce an axis at a time, but in our case
    # for some configs it may have some issue when reducing sequentially along the axes.
    group_mean = tl.sum(_sum, axis=None) / (img_size * c_per_group)
    group_var = tl.sum(_square_sum, axis=None) / (img_size * c_per_group) - group_mean * group_mean

    rstd = 1 / tl.sqrt(group_var + eps)

    # Normalize and apply linear transformation
    gamma = tl.load(gamma_ptr + cols, mask=cols < c_per_group).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=cols < c_per_group).to(tl.float32)
    for i in range(tl.cdiv(img_size, HW_SIZE)):
        x_ptr = input_ptr + i * HW_SIZE * c
        y_ptr = output_ptr + i * HW_SIZE * c
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - group_mean) * rstd
        y = x_hat * gamma + beta
        if ACTIVATION_SWISH:
            y *= tl.sigmoid(y)
        tl.store(y_ptr + offsets, y, mask=mask)


# We can have more combinations of blocks and hw_sizes, e.g.,
# blocks = [16, 32, 64, 128, 256, 512]
# hw_sizes = [8, 16, 32, 64, 128, 256, 512]
# but this will result in too many functions and slow down the compilation.
with_swish = [True, False]
dtypes = ["fp32", "fp16"]
blocks = [16, 32, 64, 128]
hw_sizes = [8, 16, 32, 64, 128, 256]
warps = [1, 2, 4, 8, 16]
name_pattern = "GroupNormTriton_{}_{}_b{}_hw{}_w{}"
sig_pattern = "*{},*{},*fp32,*fp32,i32,i32,i32,fp32"
group_pattern = "GroupNormTriton_{}_{}"


def get_function_table():
    func_table = []

    for swish, dtype, hw_size, warp, b in product(with_swish, dtypes, hw_sizes, warps, blocks):
        swish_suffix = "Swish" if swish else "Pass"
        name = name_pattern.format(swish_suffix, dtype, b, hw_size, warp)
        group = group_pattern.format(swish_suffix, dtype)
        sig = sig_pattern.format(dtype, dtype)
        kwargs = {
            "num_warps": warp,
            "constants": {"BLOCK_SIZE": b, "HW_SIZE": hw_size, "ACTIVATION_SWISH": int(swish)},
        }
        func_desc = {"name": name, "group": group, "func": group_norm_kernel, "sig": sig, "kwargs": kwargs}
        func_table.append(func_desc)
    return func_table


if __name__ == "__main__":
    func_table = get_function_table()
    for func_desc in func_table:
        print(func_desc)
