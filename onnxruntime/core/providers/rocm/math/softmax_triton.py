# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_f32 = row.to(tl.float32)
    # Subtract maximum for numerical stability
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)


# function_table = {'name': name, 'func': func, 'sig'=sig, kwargs={}},

dtypes = ["fp32", "fp16"]
blocks = [1024, 2048, 4096, 8192, 16384]
name_pattern = "softmax_{}_{}"
sig_pattern = "*{},*{},i32,i32,i32"
group_pattern = "softmax_{}"


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
            func_desc = {"name": name, "group": group, "func": softmax_kernel, "sig": sig, "kwargs": kwargs}
            func_table.append(func_desc)

    return func_table
