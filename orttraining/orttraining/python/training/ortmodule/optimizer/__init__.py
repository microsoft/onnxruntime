# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# __init__.py

# To make this method overriding work, please import this module at ealiest convinience for the training scripts.
# A suggested place is the entry trainer script.

from .fp16_optimizer import clip_grad_norm_fp32, check_overflow
import nvtx

# zero1 optimizer inefficient function override
try:
    # overwrite some deepspeed functions using optimized implementation above
    import deepspeed
    import torch

    # make sure the dependent torch/apex modules exist
    from apex.multi_tensor_apply import multi_tensor_applier
    import amp_C
    _ = torch._amp_foreach_non_finite_check_and_unscale_

    def get_grad_norm_opt(parameters, norm_type=2, mpu=None):
        total_norm = clip_grad_norm_fp32(parameters, norm_type,
                                        get_horizontal_model_parallel_rank=mpu.get_model_parallel_rank,
                                        get_horizontal_model_parallel_group=mpu.get_model_parallel_group)
        if total_norm == float(
                'inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    deepspeed.runtime.utils.get_grad_norm = get_grad_norm_opt

    def has_overflow_serial_opt(self, params, is_grad_list=False):
        return check_overflow(params)

    deepspeed.runtime.utils.CheckOverflow.has_overflow_serial = has_overflow_serial_opt

except Exception as error:
    # Error handling
    pass