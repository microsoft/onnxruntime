# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from itertools import product
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    X, Y, GAMMA, BETA, h, w, c, c_per_group, eps, BLOCK_SIZE: tl.constexpr, ACTIVATION_SWISH: tl.constexpr
):
    row_x = tl.program_id(0)
    row_y = tl.program_id(1)
    stride = h * w * c
    X += row_x * stride + row_y * c_per_group
    Y += row_x * stride + row_y * c_per_group
    GAMMA += row_y * c_per_group
    BETA += row_y * c_per_group

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < c_per_group

    # Calculate mean and variance
    group_mean = 0.0
    for i in range(h * w):
        x_ptr = X + i * c
        a = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        group_mean += tl.sum(a, axis=0)
    group_mean /= (h * w * c_per_group)

    group_var = 0.0
    for i in range(h * w):
        x_ptr = X + i * c
        a = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        a = tl.where(mask, a - group_mean, 0.)
        group_var += tl.sum(a * a, axis=0)
    group_var /= (h * w * c_per_group)

    rstd = 1 / tl.sqrt(group_var + eps)

    # Normalize and apply linear transformation
    for i in range(h * w):  # i: h * w
        y_ptr = Y + i * c
        x_ptr = X + i * c
        gamma = tl.load(GAMMA + cols, mask=mask).to(tl.float32)
        beta = tl.load(BETA + cols, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask).to(tl.float32)
        x_hat = (x - group_mean) * rstd
        y = x_hat * gamma + beta
        if ACTIVATION_SWISH:
            y = y * tl.sigmoid(y)
        tl.store(y_ptr + cols, y, mask=mask)


with_swish = [True, False]
dtypes = ["fp32", "fp16"]
blocks = [2 ** i for i in range(1, 8)]
name_pattern = "GroupNormTriton_{}_{}_b{}_w{}"
sig_pattern = "*{},*{},*fp32,*fp32,i32,i32,i32,i32,fp32"
group_pattern = "GroupNormTriton_{}_{}"


def get_function_table():
    func_table = []

    def get_num_warps(block_size):
        # return all powers of 2 but not greater than block_size
        return [2 ** i for i in range(0, 7) if 2 ** i <= block_size]

    for swish, dtype, b in product(with_swish, dtypes, blocks):
        for num_warps in get_num_warps(b):
            swish_suffix = "Swish" if swish else "Pass"
            name = name_pattern.format(swish_suffix, dtype, b, num_warps)
            group = group_pattern.format(swish_suffix, dtype)
            sig = sig_pattern.format(dtype, dtype)
            kwargs = {
                "num_warps": num_warps,
                "constants": {"BLOCK_SIZE": b, "ACTIVATION_SWISH": int(swish)},
            }
            func_desc = {"name": name, "group": group, "func": group_norm_kernel, "sig": sig, "kwargs": kwargs}
            func_table.append(func_desc)
    return func_table

if __name__ == "__main__":
    func_table = get_function_table()
    for func_desc in func_table:
        print(func_desc)
