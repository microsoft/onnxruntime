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

import torch
import types
from numpy import inf
from .modifier import FP16OptimizerModifier, check_overflow, check_overflow_for_grads

class DeepSpeedZeROModifier(FP16OptimizerModifier):
    def __init__(self, optimizer, **kwargs) -> None:
        super().__init__(optimizer)

    def can_be_modified(self):
        try:
            import amp_C
            _ = torch._amp_foreach_non_finite_check_and_unscale_
        except Exception as error:
            # Error handling
            return False
        has_overflow_serial_function = getattr(self._optimizer, "has_overflow_serial", None)
        get_grad_norm_direct_function = getattr(self._optimizer, "get_grad_norm_direct", None)
        has_overflow_partitioned_grads_serial = getattr(self._optimizer, "has_overflow_partitioned_grads_serial", None)
        if not has_overflow_serial_function or not callable(has_overflow_serial_function):
            return False
        if not get_grad_norm_direct_function or not callable(get_grad_norm_direct_function):
            return False
        if not has_overflow_partitioned_grads_serial or not callable(has_overflow_partitioned_grads_serial):
            return False
        return True

    def override_function(self):
        def get_grad_norm_direct(target, gradients, params, norm_type=2):
            from .multi_tensor_apply import MultiTensorApply
            multi_tensor_applier = MultiTensorApply(2048 * 32)
            import amp_C
            def is_model_parallel_parameter(p):
                return hasattr(p, 'model_parallel') and p.model_parallel

            norm_type = float(norm_type)
            if norm_type == inf:
                total_norm = max(g.data.abs().max() for g in gradients)
                total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
                torch.distributed.all_reduce(total_norm_cuda,
                                            op=torch.distributed.ReduceOp.MAX,
                                            group=target.dp_process_group)

                # Take max across all GPUs.
                target._model_parallel_all_reduce(tensor=total_norm_cuda,
                                                op=torch.distributed.ReduceOp.MAX)
                total_norm = total_norm_cuda[0].item()
            else:
                total_norm = 0.0
                grads_for_norm = []
                for g, p in zip(gradients, params):
                    if is_model_parallel_parameter(p) or (target.model_parallel_rank == 0):
                        # deepspeed original give a double type conversion here, not sure whether this is impacting some models.
                        # https://github.com/microsoft/DeepSpeed/blob/9e5c0c5c3ecabb68b7e9dffac0e9b8d167e3cab8/deepspeed/runtime/zero/stage2.py#L1501
                        # grads_for_norm.append(g.data.double())
                        grads_for_norm.append(g.data)

                if len(grads_for_norm) > 0:
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
                    # Since we will be summing across data parallel groups,
                    # we need the pow(norm-type).
                    total_norm_cuda = grad_norm ** norm_type
                else:
                    total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

                # Sum across all model parallel GPUs.
                torch.distributed.all_reduce(total_norm_cuda,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=target.dp_process_group)

                target._model_parallel_all_reduce(tensor=total_norm_cuda,
                                                op=torch.distributed.ReduceOp.SUM)

                total_norm = total_norm_cuda[0].item()**(1. / norm_type)

            if total_norm == float(
                    'inf') or total_norm == -float('inf') or total_norm != total_norm:
                total_norm = -1

            return total_norm

        def has_overflow_serial(target, params, is_grad_list=False):
            return check_overflow(params)

        def has_overflow_partitioned_grads_serial(target):
            for i in range(len(target.fp16_groups)):
                grad_data = [grad.data for grad in target.averaged_gradients[i] if grad is not None]
                if check_overflow_for_grads(grad_data):
                    return True
            return False

        self._optimizer.has_overflow_serial = types.MethodType(has_overflow_serial, self._optimizer)
        self._optimizer.get_grad_norm_direct = types.MethodType(get_grad_norm_direct, self._optimizer)
        # zero1 should not call into following function, is this a deepspeed bug?
        self._optimizer.has_overflow_partitioned_grads_serial = types.MethodType(has_overflow_partitioned_grads_serial, self._optimizer)
