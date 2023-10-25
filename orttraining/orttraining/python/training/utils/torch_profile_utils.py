# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch


def torch_nvtx_range_push(msg):
    if hasattr(torch.cuda.nvtx, "range_push"):
        return torch.cuda.nvtx.range_push(msg)


def torch_nvtx_range_pop():
    if hasattr(torch.cuda.nvtx, "range_pop"):
        return torch.cuda.nvtx.range_pop()


def nvtx_function_decorator(func):
    """Function decorator to record the start and end of NVTX range."""

    def wrapped_fn(*args, **kwargs):
        torch_nvtx_range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        torch_nvtx_range_pop()
        return ret_val

    return wrapped_fn
