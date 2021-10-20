# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Some functions/classes in this file are adapted from following sources:
# - post_backward_with_master_weights : https://github.com/NVIDIA/apex/blob/082f999a6e18a3d02306e27482cc7486dab71a50/apex/amp/_process_optimizer.py#L392
# - step : https://github.com/NVIDIA/apex/blob/082f999a6e18a3d02306e27482cc7486dab71a50/apex/amp/_process_optimizer.py#L364
# --------------------------------------------------------------------------

import types
import warnings
from ._modifier import FP16OptimizerModifier
import torch

class ApexAMPModifier(FP16OptimizerModifier):
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)
        pass

    def can_be_modified(self):
        return self.check_requirements(["_post_amp_backward", "zero_grad"],
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

            # Lanch the unscale kernels after 1, 2, 4, 8, ... iterations.
            # This help reduce the idle time for CUDA streams, which waiting for CPU handling the long loops.
            emit_iteration_num = 1
            fp32_from_fp16_param_count = 0
            #### END OF THE FASTER IMPLEMENTATION ####

            for fp16_param, fp32_param in zip(stash.all_fp16_params,
                                              stash.all_fp32_from_fp16_params):
                if fp16_param.grad is None and fp32_param.grad is not None:
                    continue
                elif fp16_param.grad is not None and fp32_param.grad is None:
                    #### THIS IS THE ORIGINAL IMPLEMENTATION ####
                    fp32_param.grad = torch.empty_like(fp32_param)
                    fp16_grads_needing_unscale.append(fp16_param.grad)
                    new_fp32_grads.append(fp32_param.grad)
                    #### END OF THE ORIGINAL IMPLEMENTATION ####

                    #### THIS IS THE FASTER IMPLEMENTATION ####
                    fp32_from_fp16_param_count += 1
                    if fp32_from_fp16_param_count >= emit_iteration_num:
                        scaler.unscale(
                            fp16_grads_needing_unscale,
                            new_fp32_grads,
                            scaler.loss_scale(),
                            models_are_masters=False)
                        fp16_grads_needing_unscale = []
                        new_fp32_grads = []
                        fp32_from_fp16_param_count = 0
                        emit_iteration_num = emit_iteration_num * 2
                    #### END OF THE FASTER IMPLEMENTATION ####
                elif fp16_param.grad is not None and fp32_param.grad is not None:
                    fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
                    preexisting_fp32_grads.append(fp32_param.grad)
                else: # fp16_param.grad is None and fp32_param.grad is None:
                    continue

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

        # Implementation adapted from https://github.com/NVIDIA/apex/blob/082f999a6e18a3d02306e27482cc7486dab71a50/apex/amp/_process_optimizer.py#L367
        def _zero_grad(self, set_to_none=True):
            # Apex amp's zero_grad does not have a way to set grads to none
            # This zero_grad adds a way for grads to be set to None for a faster implementation
            stash = self._amp_stash
            self._amp_lazy_init()

            # Zero the model grads.
            for param in stash.all_fp16_params:
                if set_to_none:
                    # Faster implementation
                    param.grad = None
                else:
                    # Apex amp's implementation
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

            for param in stash.all_fp32_from_fp32_params:
                if set_to_none:
                    # Faster implementation
                    param.grad = None
                else:
                    # Apex amp's implementation
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

            # Clear the master grads that are independent of model grads
            for param in stash.all_fp32_from_fp16_params:
                param.grad = None

        self._optimizer.zero_grad = types.MethodType(_zero_grad, self._optimizer)
