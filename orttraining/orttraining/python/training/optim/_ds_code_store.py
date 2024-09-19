# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright 2020 The Microsoft DeepSpeed Team
#
# !!!IMPORTANT: This file is a copy of the original one in DeepSpeed repo at given version,
# It is used to compare with the source code of current installed DeepSpeed during runtime.
# Please don't modify it or do any code formatting for it.
# 'orttraining/orttraining/python/training/optim/_ds_code_store.py' is removed from lintrunner config by intention.
# --------------------------------------------------------------------------

# Wrap code in this to make sure the indentation is correct compared with raw DeepSpeed.

class Stage1And2_DeepSpeedZeroOptimizer_0_9_2:

    def has_overflow_serial(self, params, is_grad_list=False):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False


    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
        else:
            total_norm = 0.0
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm beginning {total_norm}")
            for g, p in zip(gradients, params):
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item()**2
            # Sum across all model parallel GPUs.
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

            total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm


    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False
