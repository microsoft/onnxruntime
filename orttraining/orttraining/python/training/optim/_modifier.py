# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Some functions in this file are adapted from following sources:
# - clip_grad_norm_fp32 : https://github.com/NVIDIA/Megatron-LM/blob/5ac5571ba0265af4c491ee0af1508ca7589450c6/megatron/optimizer/clip_grads.py#L29
# - check_overflow_for_grads : https://github.com/NVIDIA/Megatron-LM/blob/5ac5571ba0265af4c491ee0af1508ca7589450c6/megatron/optimizer/optimizer.py#L341
# --------------------------------------------------------------------------

import torch
from numpy import inf
from ._multi_tensor_apply import MultiTensorApply
multi_tensor_applier = MultiTensorApply(2048 * 32)

class FP16OptimizerModifier(object):
    def __init__(self, optimizer) -> None:
        super().__init__()
        self._optimizer = optimizer

    def apply(self):
        if self.can_be_modified():
            self.override_function()

    def check_requirements(self, required_funcs, require_apex=False, require_torch_non_finite_check=False):
        try:
            if require_apex is True:
                import amp_C
            if require_torch_non_finite_check is True:
                _ = torch._amp_foreach_non_finite_check_and_unscale_
        except Exception as _:
            return False

        if required_funcs:
            for func_name in required_funcs:
                func = getattr(self._optimizer, func_name, None)
                if not func or not callable(func):
                    return False
        return True

def check_overflow(params):
    grad_data = [p.grad.data for p in params if p.grad is not None]
    return check_overflow_for_grads(grad_data) 

def check_overflow_for_grads(grad_data):
    found_inf = torch.cuda.FloatTensor([0.0])
    scaler = torch.cuda.FloatTensor([1.0])
    # Unscale and set found inf/nan
    torch._amp_foreach_non_finite_check_and_unscale_(grad_data, found_inf, scaler)

    # Check for nan.
    overflow = (found_inf.item() > 0)
    return overflow 

def clip_grad_norm_fp32(parameters, max_norm, norm_type,
                        get_horizontal_model_parallel_rank=None, get_horizontal_model_parallel_group=None):
    import amp_C

    horizontal_model_parallel_grad_norm_aggregation = False
    if get_horizontal_model_parallel_rank and get_horizontal_model_parallel_group:
        horizontal_model_parallel_grad_norm_aggregation = True

    def param_is_not_tensor_parallel_duplicate(param):
        is_mp_tensor = hasattr(param, 'model_parallel') and param.model_parallel
        return is_mp_tensor or (get_horizontal_model_parallel_rank() == 0)

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - should not be a replica due to tensor model parallelism
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        grad = param.grad.detach()
        if grad_not_none:
            # Make sure the grads are in fp32
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            if horizontal_model_parallel_grad_norm_aggregation:
                is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
                if grad_not_none and is_not_tp_duplicate:
                    grads_for_norm.append(grad)
            else:
                grads_for_norm.append(grad)

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        if horizontal_model_parallel_grad_norm_aggregation:
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

            # Take max across all model-parallel GPUs.
            torch.distributed.all_reduce(total_norm_cuda,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=get_horizontal_model_parallel_group())
            total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            grad_norm, _ = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads_for_norm],
                False # no per-parameter norm
            )

            if not horizontal_model_parallel_grad_norm_aggregation:
                return grad_norm.item()

            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        if horizontal_model_parallel_grad_norm_aggregation:
            # Sum across all model-parallel GPUs.
            torch.distributed.all_reduce(total_norm,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=get_horizontal_model_parallel_group())
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm

