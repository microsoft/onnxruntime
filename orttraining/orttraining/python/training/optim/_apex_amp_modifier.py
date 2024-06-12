# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Some functions in this file are adapted from NVIDIA/apex, commit 082f999a6e18a3d02306e27482cc7486dab71a50
# --------------------------------------------------------------------------

import types
import warnings

from ._modifier import FP16OptimizerModifier


class ApexAMPModifier(FP16OptimizerModifier):
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)

    def can_be_modified(self):
        return self.check_requirements(
            ["_post_amp_backward", "zero_grad"], require_apex=True, require_torch_non_finite_check=False
        )

    def override_function(m_self):  # noqa: N805
        from apex import amp as apex_amp

        from onnxruntime.training.ortmodule.torch_cpp_extensions import fused_ops

        warnings.warn("Apex AMP fp16_optimizer functions are overrided with faster implementation.", UserWarning)

        # Implementation adapted from https://github.com/NVIDIA/apex/blob/082f999a6e18a3d02306e27482cc7486dab71a50/apex/amp/_process_optimizer.py#L161
        def post_backward_with_master_weights(self, scaler):
            stash = self._amp_stash

            self._amp_lazy_init()

            #### THIS IS THE ORIGINAL IMPLEMENTATION ####
            # # This is a lot of python overhead...
            # fp16_grads_needing_unscale = []
            # new_fp32_grads = []
            # fp16_grads_needing_unscale_with_stash = []
            # preexisting_fp32_grads = [
            # i = 0
            # for fp16_param, fp32_param in zip(stash.all_fp16_params,
            #                                  stash.all_fp32_from_fp16_params):
            #     if fp16_param.grad is None and fp32_param.grad is not None:
            #         continue
            #     if fp16_param.grad is not None and fp32_param.grad is None:
            #         print(i, " shape ", fp32_param.shape)
            #         i += 1
            #         fp32_param.grad = torch.empty_like(fp32_param)
            #         fp16_grads_needing_unscale.append(fp16_param.grad)
            #         new_fp32_grads.append(fp32_param.grad)
            #     elif fp16_param.grad is not None and fp32_param.grad is not None:
            #         fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
            #         preexisting_fp32_grads.append(fp32_param.grad)
            #     else: # fp16_param.grad is None and fp32_param.grad is None:
            #         continue

            # if len(fp16_grads_needing_unscale) > 0:
            #     scaler.unscale(
            #         fp16_grads_needing_unscale,
            #         new_fp32_grads,
            #         scaler.loss_scale(),
            #         models_are_masters=False)

            # if len(fp16_grads_needing_unscale_with_stash) > 0:
            #     scaler.unscale_with_stashed(
            #         fp16_grads_needing_unscale_with_stash,
            #         preexisting_fp32_grads,
            #         preexisting_fp32_grads)
            #### END OF THE ORIGINAL IMPLEMENTATION ####

            #### THIS IS THE FASTER IMPLEMENTATION ####
            tensor_vector_exist = hasattr(stash, "all_fp16_params_tensor_vector") and hasattr(
                stash, "all_fp32_from_fp16_params_tensor_vector"
            )
            tensor_vector_valid = (
                tensor_vector_exist
                and len(stash.all_fp16_params_tensor_vector) == len(stash.all_fp16_params)
                and len(stash.all_fp32_from_fp16_params_tensor_vector) == len(stash.all_fp32_from_fp16_params)
            )

            if not tensor_vector_valid:
                stash.all_fp16_params_tensor_vector = fused_ops.TorchTensorVector(stash.all_fp16_params)
                stash.all_fp32_from_fp16_params_tensor_vector = fused_ops.TorchTensorVector(
                    stash.all_fp32_from_fp16_params
                )

            fused_ops.unscale_fp16_grads_into_fp32_grads(
                stash.all_fp16_params_tensor_vector,
                stash.all_fp32_from_fp16_params_tensor_vector,
                scaler._overflow_buf,
                scaler._loss_scale,
            )
            #### END OF THE FASTER IMPLEMENTATION ####
            # fp32 params can be treated as they would be in the "no_master_weights" case.
            apex_amp._process_optimizer.post_backward_models_are_masters(
                scaler, stash.all_fp32_from_fp32_params, stash.all_fp32_from_fp32_grad_stash
            )

        from apex.optimizers import FusedSGD

        if not isinstance(m_self._optimizer, FusedSGD):
            m_self._optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights, m_self._optimizer
            )

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

        m_self._optimizer.zero_grad = types.MethodType(_zero_grad, m_self._optimizer)
