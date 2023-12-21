# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import subprocess as sp

import torch


def torch_nvtx_range_push(msg):
    if hasattr(torch.cuda.nvtx, "range_push"):
        torch.cuda.nvtx.range_push(msg)


def torch_nvtx_range_pop():
    if hasattr(torch.cuda.nvtx, "range_pop"):
        torch.cuda.nvtx.range_pop()


def nvtx_function_decorator(func):
    """Function decorator to record the start and end of NVTX range."""

    def wrapped_fn(*args, **kwargs):
        torch_nvtx_range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        torch_nvtx_range_pop()
        return ret_val

    return wrapped_fn


def log_memory_usage(cur_phase: str, rank_0_only=True, step_info="", logger=None, module=None):
    """Log memory usage for the current phase.
    Args:
        cur_phase (str): The current phase.
        rank_0_only (bool, optional): Only log the memory usage for rank 0. Defaults to True.
        step_info (str, optional): The step information. Defaults to "".
        logger (logging.Logger, optional): The logger to log the memory usage. Defaults to None, which means print to stdout.
        module (torch.nn.Module, optional): The module to get parameter, buffer and grad sizes. Defaults to None.
    """
    rank = 0
    if rank_0_only is True:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank != 0:
            return

    _normalizer_factor = float(1024 * 1024)
    _normalizer_unit = "MiB"

    def _normalize(mem_size_in_bytes: float | int) -> str:
        return f"{float(mem_size_in_bytes) / _normalizer_factor:.0f}"

    cur_mem_allocated = _normalize(torch.cuda.memory_allocated())
    max_mem_allocated = _normalize(torch.cuda.max_memory_allocated())
    cur_mem_cached = _normalize(torch.cuda.memory_reserved())
    max_mem_cached = _normalize(torch.cuda.max_memory_reserved())
    torch_mem_stat = torch.cuda.memory_stats()
    cur_mem_inactive = _normalize(torch_mem_stat.get("inactive_split_bytes.all.current", 0))
    max_mem_inactive = _normalize(torch_mem_stat.get("inactive_split_bytes.all.peak", 0))

    def output_to_list(x):
        return x.decode("ascii").split("\n")[:-1]

    nvm_cmd = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(nvm_cmd.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError(f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}") from None
    memory_use_value = [int(x.split()[0]) for i, x in enumerate(memory_use_info)][rank]

    mem_stats = [
        ["phase", cur_phase],
        ["nvm smi", memory_use_value],
        ["allocated", cur_mem_allocated],  # current memory allocated for tensors
        ["max allocated", max_mem_allocated],  # peak memory allocated for tensors
        ["cached", cur_mem_cached],  # current memory cached for the caching allocator
        ["max cached", max_mem_cached],  # peak memory cached for caching allocator.
        ["inactive", cur_mem_inactive],  # amount of inactive, non-releasable memory
        ["max inactive", max_mem_inactive],  # peak of inactive, non-releasable memory
    ]

    # Calculate the total size of parameters and gradients in the model
    if module:
        param_total_size = 0
        grad_total_size = 0
        for p in module.parameters():
            if p.is_cuda:
                param_total_size += p.numel() * p.element_size()
            if p.grad is not None and p.grad.is_cuda:
                grad_total_size += p.grad.numel() * p.grad.element_size()

        # Calculate the total size of buffers in the model
        buffer_total_size = 0
        for b in module.buffers():
            if b.is_cuda:
                buffer_total_size += b.numel() * b.element_size()

        mem_stats.extend(
            [
                ["param size", _normalize(param_total_size)],
                ["grad size", _normalize(grad_total_size)],
                ["buffer size", _normalize(buffer_total_size)],
            ]
        )

    summ = f"rank-{rank} {step_info} memory ({_normalizer_unit})"
    for stat in mem_stats:
        summ += f" | {stat[0]}: {stat[1]}"

    if logger is None:
        print(summ)
    else:
        logger.info(summ)
