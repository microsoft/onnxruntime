# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright 2020 The Microsoft DeepSpeed Team
# Copyright (c) 2020, NVIDIA CORPORATION.
# Some functions in this file are adapted from following sources:
# - _check_overflow : https://github.com/microsoft/DeepSpeedExamples/blob/590364d482b592c3a8a44c28141a8139c7918c55/Megatron-LM-v1.1.5-ZeRO3/megatron/fp16/fp16.py#L294
# - clip_master_grads : https://github.com/microsoft/DeepSpeedExamples/blob/590364d482b592c3a8a44c28141a8139c7918c55/Megatron-LM-v1.1.5-ZeRO3/megatron/fp16/fp16.py#L332
# --------------------------------------------------------------------------

import types
import warnings

from ._modifier import FP16OptimizerModifier, check_overflow, clip_grad_norm_fp32


class LegacyMegatronLMModifier(FP16OptimizerModifier):
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)
        self.get_horizontal_model_parallel_rank = kwargs.get("get_horizontal_model_parallel_rank", None)
        self.get_horizontal_model_parallel_group = kwargs.get("get_horizontal_model_parallel_group", None)

    def can_be_modified(self):
        return self.check_requirements(
            ["_check_overflow", "clip_master_grads"], require_apex=False, require_torch_non_finite_check=True
        )

    def override_function(self):
        warnings.warn("Megatron-LM fp16_optimizer functions are overrided with faster implementation.", UserWarning)

        def clip_master_grads(target, max_norm, norm_type=2):
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
            if not target.overflow:
                fp32_params = []
                for param_group in target.optimizer.param_groups:
                    for param in param_group["params"]:
                        fp32_params.append(param)  # noqa: PERF402
                #### THIS IS THE ORIGINAL IMPLEMENTATION ####
                # return self.clip_grad_norm(fp32_params, max_norm, norm_type)
                #### END OF THE ORIGINAL IMPLEMENTATION ####

                #### THIS IS THE FASTER IMPLEMENTATION ####
                return clip_grad_norm_fp32(
                    fp32_params,
                    max_norm,
                    norm_type,
                    get_horizontal_model_parallel_rank=self.get_horizontal_model_parallel_rank,
                    get_horizontal_model_parallel_group=self.get_horizontal_model_parallel_group,
                )
                #### END OF THE FASTER IMPLEMENTATION ####
            else:
                return -1

        def _check_overflow(target):
            params = []
            for group in target.fp16_groups:
                for param in group:
                    params.append(param)  # noqa: PERF402
            for group in target.fp32_from_fp32_groups:
                for param in group:
                    params.append(param)  # noqa: PERF402
            #### THIS IS THE ORIGINAL IMPLEMENTATION ####
            # self.overflow = self.loss_scaler.has_overflow(params)
            #### END OF THE ORIGINAL IMPLEMENTATION ####

            #### THIS IS THE FASTER IMPLEMENTATION ####
            target.overflow = check_overflow(params)
            #### END OF THE FASTER IMPLEMENTATION ####
            return target.overflow

        self._optimizer._check_overflow = types.MethodType(_check_overflow, self._optimizer)
        self._optimizer.clip_master_grads = types.MethodType(clip_master_grads, self._optimizer)
