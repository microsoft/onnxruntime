# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright 2020 The Microsoft DeepSpeed Team
# Copyright (c) 2020, NVIDIA CORPORATION.
# Some functions in this file are adapted from following sources:
# - has_overflow_serial : https://github.com/microsoft/DeepSpeed/blob/d8e9ef6f99e27bb95e10bd146d145b3372b4cfda/deepspeed/runtime/zero/stage2.py#L1792
# - get_grad_norm_direct : https://github.com/microsoft/DeepSpeed/blob/d8e9ef6f99e27bb95e10bd146d145b3372b4cfda/deepspeed/runtime/zero/stage2.py#L1466
# - has_overflow_partitioned_grads_serial : https://github.com/microsoft/DeepSpeed/blob/d8e9ef6f99e27bb95e10bd146d145b3372b4cfda/deepspeed/runtime/zero/stage2.py#L1799
# --------------------------------------------------------------------------

import types
import warnings

import torch
from numpy import inf
from packaging.version import Version

from ._modifier import FP16OptimizerModifier, check_overflow, check_overflow_for_grads
from ._multi_tensor_apply import MultiTensorApply

multi_tensor_applier = MultiTensorApply(2048 * 32)


class DeepSpeedZeROModifier(FP16OptimizerModifier):
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)

    def can_be_modified(self):
        import deepspeed

        # This modifier relies on the implementation of has_overflow_serial, get_grad_norm_direct,
        # and has_overflow_partitioned_grads_serial
        # in https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py.
        # Everytime if we want to update this version supporting list to a newer version,
        # we need to check if the implementation of these functions are changed.
        # An easy way to check is to check the history of this file, if there is no change during the update,
        # it's safe to update the version supporting list. Otherwise, or the file is moved or renamed,
        # we need to check the implementation of these functions in detail.
        ds_version = Version(deepspeed.__version__)
        if ds_version > Version("0.9.1") or ds_version < Version("0.4.0"):
            warnings.warn(
                "Skip modifying optimizer because of unsupported DeepSpeed version {}, "
                "supported version: 0.4.0 - 0.9.1.".format(deepspeed.__version__),
                UserWarning,
            )
            return False

        try:
            from deepspeed.accelerator import get_accelerator
        except ImportError:
            warnings.warn("Unable to import get_accelerator from deepspeed.accelerator", UserWarning)
        else:
            if not get_accelerator().device_name().startswith("cuda"):
                warnings.warn(
                    "Skip modifying optimizer as device is not supported, "
                    f"device name: {get_accelerator().device_name()}",
                    UserWarning,
                )
                return False

        return self.check_requirements(
            ["has_overflow_serial", "get_grad_norm_direct", "has_overflow_partitioned_grads_serial"],
            require_apex=False,
            require_torch_non_finite_check=True,
        )

    def override_function(self):
        warnings.warn("DeepSpeed fp16_optimizer functions are overrided with faster implementation.", UserWarning)

        def get_grad_norm_direct(target, gradients, params, norm_type=2):
            from onnxruntime.training.ortmodule.torch_cpp_extensions import fused_ops

            def is_model_parallel_parameter(p):
                return hasattr(p, "model_parallel") and p.model_parallel

            norm_type = float(norm_type)
            if norm_type == inf:
                total_norm = max(g.data.abs().max() for g in gradients)
                total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
                torch.distributed.all_reduce(
                    total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=target.dp_process_group
                )

                # Take max across all GPUs.
                target._model_parallel_all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.MAX)
                total_norm = total_norm_cuda[0].item()
            else:
                total_norm = 0.0

                #### THIS IS THE ORIGINAL IMPLEMENTATION ####
                # # if dist.get_rank() == 0:
                # #    logger.info(f"Total Norm beginning {total_norm}")
                # for g, p in zip(gradients, params):
                #     # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                #     if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                #         continue
                #     if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                #         param_norm = g.data.double().norm(2)
                #         total_norm += param_norm.item()**2
                # # Sum across all model parallel GPUs.
                # total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
                #### END OF THE ORIGINAL IMPLEMENTATION ####

                #### THIS IS THE FASTER IMPLEMENTATION ####
                grads_for_norm = []
                for g, p in zip(gradients, params):
                    if is_model_parallel_parameter(p) or (target.model_parallel_rank == 0):
                        # BE NOTED: deepspeed original give a double type conversion here, not sure whether this is impacting some models.
                        # https://github.com/microsoft/DeepSpeed/blob/9e5c0c5c3ecabb68b7e9dffac0e9b8d167e3cab8/deepspeed/runtime/zero/stage2.py#L1501
                        # grads_for_norm.append(g.data.double())
                        grads_for_norm.append(g.data)

                if len(grads_for_norm) > 0:
                    dummy_overflow_buf = torch.cuda.IntTensor([0])
                    # Use apex's multi-tensor applier for efficiency reasons.
                    # Multi-tensor applier takes a function and a list of list
                    # and performs the operation on that list all in one kernel.
                    grad_norm, _ = multi_tensor_applier(
                        fused_ops.multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [fused_ops.TorchTensorVector(grads_for_norm)],
                        False,  # no per-parameter norm
                    )
                    # Since we will be summing across data parallel groups,
                    # we need the pow(norm-type).
                    total_norm_cuda = grad_norm**norm_type
                else:
                    total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
                #### END OF THE FASTER IMPLEMENTATION ####

                # Sum across all model parallel GPUs.
                torch.distributed.all_reduce(
                    total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=target.dp_process_group
                )

                target._model_parallel_all_reduce(tensor=total_norm_cuda, op=torch.distributed.ReduceOp.SUM)

                total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)

            if total_norm == float("inf") or total_norm == -float("inf") or total_norm != total_norm:
                total_norm = -1

            return total_norm

        def has_overflow_serial(target, params, is_grad_list=False):
            #### THIS IS THE ORIGINAL IMPLEMENTATION ####
            # for p in params:
            #     if p.grad is not None and self._has_inf_or_nan(p.grad.data):
            #         return True
            #
            # return False
            #### END OF THE ORIGINAL IMPLEMENTATION ####

            #### THIS IS THE FASTER IMPLEMENTATION ####
            return check_overflow(params)
            #### END OF THE FASTER IMPLEMENTATION ####

        def has_overflow_partitioned_grads_serial(target):
            #### THIS IS THE ORIGINAL IMPLEMENTATION ####
            # for i in range(len(self.fp16_groups)):
            #     for j, grad in enumerate(self.averaged_gradients[i]):
            #         if grad is not None and self._has_inf_or_nan(grad.data, j):
            #             return True
            # return False
            #### END OF THE ORIGINAL IMPLEMENTATION ####

            #### THIS IS THE FASTER IMPLEMENTATION ####
            groups = target.fp16_groups if hasattr(target, "fp16_groups") else target.bit16_groups
            for i in range(len(groups)):
                grad_data = [grad.data for grad in target.averaged_gradients[i] if grad is not None]
                if check_overflow_for_grads(grad_data):
                    return True
            return False
            #### END OF THE FASTER IMPLEMENTATION ####

        self._optimizer.has_overflow_serial = types.MethodType(has_overflow_serial, self._optimizer)
        self._optimizer.get_grad_norm_direct = types.MethodType(get_grad_norm_direct, self._optimizer)
        # zero1 should not call into following function, is this a deepspeed bug?
        self._optimizer.has_overflow_partitioned_grads_serial = types.MethodType(
            has_overflow_partitioned_grads_serial, self._optimizer
        )
