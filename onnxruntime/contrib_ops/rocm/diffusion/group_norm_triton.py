# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(X, Y, gamma, beta, h, w, c, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    stride = h * w * c
    X += row * stride
    Y += row * stride

    # BLOCK_SIZE: c / g
    _group_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(h * w):  # i: h * w
        x_ptr = X + i * c
        for off in range(0, c, BLOCK_SIZE):  # off: group
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(x_ptr + cols, mask=cols < c, other=0.).to(tl.float32)
            _group_mean += a
    group_mean = _group_mean / (h * w * c / BLOCK_SIZE)

    _group_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(h * w):  # i: h * w
        x_ptr = X + i * c
        for off in range(0, c, BLOCK_SIZE):  # off: group
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(x_ptr + cols, mask=cols < c, other=0.).to(tl.float32)
            a = tl.where(cols < c, a - group_mean, 0.)
            _group_var += a * a
    group_var = _group_var / (h * w * c / BLOCK_SIZE)

    rstd = 1 / tl.sqrt(group_var + eps)

    # Normalize and apply linear transformation
    for i in range(h * w):  # i: h * w
        y_ptr = Y + i * c
        x_ptr = X + i * c
        for off in range(0, c, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < c
            gamma = tl.load(gamma + cols, mask=mask).to(tl.float32)
            beta = tl.load(beta + cols, mask=mask).to(tl.float32)
            x = tl.load(x_ptr + cols, mask=mask).to(tl.float32)
            x_hat = (x - group_mean) * rstd
            y = x_hat * gamma + beta
            tl.store(y_ptr + cols, y, mask=mask)



dtypes = ["fp32", "fp16"]
blocks = [32]
name_pattern = "group_norm_{}_{}"
sig_pattern = "*{},*{},i32,i32,i32"
group_pattern = "group_norm_{}"


def get_function_table():
    func_table = []

    def get_num_warps(block_size):
        num_warps = 4
        if block_size >= 2048:
            num_warps = 8
        if block_size >= 4096:
            num_warps = 16
        return num_warps

    for dtype in dtypes:
        for b in blocks:
            name = name_pattern.format(dtype, b)
            group = group_pattern.format(dtype)
            sig = sig_pattern.format(dtype, dtype)
            num_warps = get_num_warps(b)
            kwargs = {"num_warps": num_warps, "constants": {"BLOCK_SIZE": b}}
            func_desc = {"name": name, "group": group, "func": group_norm_kernel, "sig": sig, "kwargs": kwargs}
            func_table.append(func_desc)

    return func_table
