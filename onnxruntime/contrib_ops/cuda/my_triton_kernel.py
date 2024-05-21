# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def my_triton_kernel(output_ptr, input_ptr, N0, BLOCK_SIZE: tl.constexpr):
    pid_0 = tl.program_id(0)
    block_start = BLOCK_SIZE * pid_0
    block_range = block_start + tl.arange(0, BLOCK_SIZE)

    row = tl.load(input_ptr + block_range, block_range < N0)
    tl.store(output_ptr + block_range, -row, block_range < N0)


dtypes = ["fp32", "fp16"]
blocks = [64, 1024, 2048]
name_pattern = "my_triton_kernel_{}_{}"
sig_pattern = "*{},*{},i32"
group_pattern = "my_triton_kernel_{}"


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
            func_desc = {"name": name, "group": group, "func": my_triton_kernel, "sig": sig, "kwargs": kwargs}
            func_table.append(func_desc)

    return func_table
