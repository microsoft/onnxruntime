# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import torch
import types
from numpy import inf

def megatron_lm_check(optimizer):
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
        import amp_C
        _ = torch._amp_foreach_non_finite_check_and_unscale_
    except Exception as error:
        # Error handling
        return False
    _check_overflow_function = getattr(optimizer, "_check_overflow", None)
    clip_master_grads_function = getattr(optimizer, "clip_master_grads", None)
    if not _check_overflow_function or not callable(_check_overflow_function):
        return False
    if not clip_master_grads_function or not callable(clip_master_grads_function):
        return False
    return True

def FP16_Optimizer(optimizer, **kwargs):
    """
    Simple wrapper to replace inefficient FP16_Optimizer function calls implemented by library for example
        Apex, DeepSpeed, Megatron-LM.

    Args:
        optimizer: the FP16_Optimizer instance

    Returns:
        the FP16_Optimizer instance

    """
    def get_full_qualified_type_name(o):
        klass = o.__class__
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__ # avoid outputs like 'builtins.str'
        return module + '.' + klass.__qualname__

    optimizer_full_name = get_full_qualified_type_name(optimizer)
    if optimizer_full_name == "megatron.fp16.fp16.FP16_Optimizer":
        if megatron_lm_check(optimizer):
            # get_horizontal_model_parallel_rank: function to get horizontal model parallel rank
            # get_horizontal_model_parallel_group: function to get horizontal model parallel group.
            get_horizontal_model_parallel_rank = kwargs.get("get_horizontal_model_parallel_rank", None)
            get_horizontal_model_parallel_group = kwargs.get("get_horizontal_model_parallel_group", None)
            def clip_master_grads(self, max_norm, norm_type=2):
                """
                Clips fp32 master gradients via ``torch.nn.utils.clip_grad_norm``.

                Args:
                    max_norm (float or int): max norm of the gradients
                    norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                        infinity norm.

                Returns:
                    Total norm of the current fp32 gradients (viewed as a single vector).

                .. warning::
                    Returns -1 if the most recently computed fp16 gradients overflowed (that is, if ``self.overflow`` is ``True``).
                """
                if not self.overflow:
                    print("clip_grad_norm_fp32 called....................................")
                    fp32_params = []
                    for param_group in self.optimizer.param_groups:
                        for param in param_group['params']:
                            fp32_params.append(param)
                    return clip_grad_norm_fp32(fp32_params, max_norm, norm_type, 
                                               get_horizontal_model_parallel_rank=get_horizontal_model_parallel_rank,
                                               get_horizontal_model_parallel_group=get_horizontal_model_parallel_group)
                else:
                    return -1

            def _check_overflow(self):
                params = []
                for group in self.fp16_groups:
                    for param in group:
                        params.append(param)
                for group in self.fp32_from_fp32_groups:
                    for param in group:
                        params.append(param)
                self.overflow = check_overflow(params)
                return self.overflow

            optimizer._check_overflow = types.MethodType(_check_overflow, optimizer)
            optimizer.clip_master_grads = types.MethodType(clip_master_grads, optimizer)
        else:
            print("megatron.fp16.fp16.FP16_Optimizer wrapping failed.")

    return optimizer

def check_overflow(params):
    found_inf = torch.cuda.FloatTensor([0.0])
    scaler = torch.cuda.FloatTensor([1.0])
    grad_data = [p.grad.data for p in params if p.grad is not None]
    # Unscale and set found inf/nan
    torch._amp_foreach_non_finite_check_and_unscale_(grad_data, found_inf, scaler)

    # Check for nan.
    overflow = (found_inf.item() > 0)
    return overflow 


def clip_grad_norm_fp32(parameters, max_norm, norm_type,
                        get_horizontal_model_parallel_rank=None, get_horizontal_model_parallel_group=None):
    from apex.multi_tensor_apply import multi_tensor_applier
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
