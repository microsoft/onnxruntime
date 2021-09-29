# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
import types
from .fp16_optimizer import FP16OptimizerModifier, check_overflow, clip_grad_norm_fp32

class LegacyMegatronLMModifier(FP16OptimizerModifier):
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)
        self.get_horizontal_model_parallel_rank = kwargs.get("get_horizontal_model_parallel_rank", None)
        self.get_horizontal_model_parallel_group = kwargs.get("get_horizontal_model_parallel_group", None)

    def can_be_modified(self):
        try:
            from apex.multi_tensor_apply import multi_tensor_applier
            import amp_C
            _ = torch._amp_foreach_non_finite_check_and_unscale_
        except Exception as error:
            # Error handling
            return False
        _check_overflow_function = getattr(self.optimizer, "_check_overflow", None)
        clip_master_grads_function = getattr(self.optimizer, "clip_master_grads", None)
        if not _check_overflow_function or not callable(_check_overflow_function):
            return False
        if not clip_master_grads_function or not callable(clip_master_grads_function):
            return False
        return True

    def override_function(self):
        def clip_master_grads(self_, max_norm, norm_type=2):
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
            if not self_.overflow:
                fp32_params = []
                for param_group in self_.optimizer.param_groups:
                    for param in param_group['params']:
                        fp32_params.append(param)
                return clip_grad_norm_fp32(fp32_params, max_norm, norm_type, 
                                            get_horizontal_model_parallel_rank=self.get_horizontal_model_parallel_rank,
                                            get_horizontal_model_parallel_group=self.get_horizontal_model_parallel_group)
            else:
                return -1

        def _check_overflow(self_):
            params = []
            for group in self_.fp16_groups:
                for param in group:
                    params.append(param)
            for group in self_.fp32_from_fp32_groups:
                for param in group:
                    params.append(param)
            self_.overflow = check_overflow(params)
            return self_.overflow

        self.optimizer._check_overflow = types.MethodType(_check_overflow, self.optimizer)
        self.optimizer.clip_master_grads = types.MethodType(clip_master_grads, self.optimizer)
