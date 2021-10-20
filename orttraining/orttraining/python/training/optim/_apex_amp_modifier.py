# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Some functions/classes in this file are adapted from following sources:
# - post_backward_with_master_weights : https://github.com/NVIDIA/apex/blob/082f999a6e18a3d02306e27482cc7486dab71a50/apex/amp/_process_optimizer.py#L392
# - step : https://github.com/NVIDIA/apex/blob/082f999a6e18a3d02306e27482cc7486dab71a50/apex/amp/_process_optimizer.py#L364
# MemoryBuffer - https://github.com/NVIDIA/Megatron-LM/blob/aed2f75e209e525c842aec7c044af7acae2a4614/megatron/model/distributed.py#L28
# --------------------------------------------------------------------------

import types
import warnings
from ._modifier import FP16OptimizerModifier
import torch

class MemoryBuffer:
    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.empty(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)

    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the 1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

class ApexAMPModifier(FP16OptimizerModifier):
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)
        pass

    def can_be_modified(self):
        return self.check_requirements(["_post_amp_backward", "step"],
                                       require_apex=True, require_torch_non_finite_check=False)

    def override_function(self):
        from apex import amp as apex_amp
        warnings.warn('Apex AMP fp16_optimizer functions are overrided with faster implementation.', UserWarning)
        def post_backward_with_master_weights(self, scaler):
            stash = self._amp_stash
            self._amp_lazy_init()

            # This is a lot of python overhead...
            fp16_grads_needing_unscale = []
            new_fp32_grads = []
            fp16_grads_needing_unscale_with_stash = []
            preexisting_fp32_grads = []

            #### THIS IS THE FASTER IMPLEMENTATION ####
            fp32_from_fp16_param_next_offset = 0
            fp32_from_fp16_params = []
            #### END OF THE FASTER IMPLEMENTATION ####

            for fp16_param, fp32_param in zip(stash.all_fp16_params,
                                                stash.all_fp32_from_fp16_params):
                if fp16_param.grad is None and fp32_param.grad is not None:
                    continue
                elif fp16_param.grad is not None and fp32_param.grad is None:
                    #### THIS IS THE ORIGINAL IMPLEMENTATION ####
                    # fp32_param.grad = torch.empty_like(fp32_param)
                    fp16_grads_needing_unscale.append(fp16_param.grad)
                    # new_fp32_grads.append(fp32_param.grad)
                    #### END OF THE ORIGINAL IMPLEMENTATION ####

                    #### THIS IS THE FASTER IMPLEMENTATION ####
                    fp32_from_fp16_params.append(fp32_param)
                    fp32_from_fp16_param_next_offset += fp32_param.data.nelement()
                    #### END OF THE FASTER IMPLEMENTATION ####
                elif fp16_param.grad is not None and fp32_param.grad is not None:
                    fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
                    preexisting_fp32_grads.append(fp32_param.grad)
                else: # fp16_param.grad is None and fp32_param.grad is None:
                    continue

            #### THIS IS THE FASTER IMPLEMENTATION ####
            # allocate the buffer at once.
            stash._fp32_from_fp16_param_grad_buffers = MemoryBuffer(fp32_from_fp16_param_next_offset, torch.float)

            fp32_from_fp16_param_next_offset = 0
            for fp32_param in fp32_from_fp16_params:
                fp32_param.grad = stash._fp32_from_fp16_param_grad_buffers.get(fp32_param.data.shape, fp32_from_fp16_param_next_offset)
                new_fp32_grads.append(fp32_param.grad)
                fp32_from_fp16_param_next_offset += fp32_param.data.nelement()
            #### END OF THE FASTER IMPLEMENTATION ####

            if len(fp16_grads_needing_unscale) > 0:
                scaler.unscale(
                    fp16_grads_needing_unscale,
                    new_fp32_grads,
                    scaler.loss_scale(),
                    models_are_masters=False)

            if len(fp16_grads_needing_unscale_with_stash) > 0:
                scaler.unscale_with_stashed(
                    fp16_grads_needing_unscale_with_stash,
                    preexisting_fp32_grads,
                    preexisting_fp32_grads)

            # fp32 params can be treated as they would be in the "no_master_weights" case.
            #post_backward_models_are_masters()
            apex_amp._process_optimizer.post_backward_models_are_masters(
                scaler,
                stash.all_fp32_from_fp32_params,
                stash.all_fp32_from_fp32_grad_stash)

        from apex.optimizers import FusedSGD as FusedSGD
        if not isinstance(self._optimizer, FusedSGD):
            self._optimizer._post_amp_backward = types.MethodType(post_backward_with_master_weights, self._optimizer)

        old_step = self._optimizer.step
        def updated_step(self, closure=None):
            retval = old_step()
            # remove the allocation for fp32_from_fp16_param_grad buffer
            self._amp_stash._fp32_from_fp16_param_grad_buffers = None
            return retval

        self._optimizer.step = types.MethodType(updated_step, self._optimizer)
