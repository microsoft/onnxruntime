# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

import torch

from ._subscriber_base import SubscriberBase, _ORTTensorInspector, _RuntimeStates


class StatisticsSubscriber(SubscriberBase):
    """
    This subscriber is used to dump the activation statistics into files.

    Each activation will be summarized into 1 or 2 files, depending on whether it is used in the backward pass.
    > In the forward pass, summarize the tensor's statistics and write to a file.
    > In the backward pass, summarize the tensor's gradient statistics and write it into another file.
    So for each run step, there will be many files.

    Currently, the statistics mainly include:
    > Number of inf/nan values.
    > Common statistics: tensor shape, data type, total element size, min/max/mean/std of the tensor elements.
    > Number of zero/negative/positive elements.
    > Sampled elements (taking the first 128 elements for example).
    > To be extended...

    `merge_activation_summary.py` can be used to merge the files into one file per training step.
    """

    def __init__(
        self,
        output_dir: str,
        start_step: Union[None, int] = None,
        end_step: Union[None, int] = None,
        override_output_dir: bool = False,
        run_on_cpu: bool = False,
        bucket_size: int = 1024 * 1024 * 1024 // 2,
    ):
        """
        Steps in [start_step, end_step) will run subscriber actions.

        Args:
            output_dir: the directory in all activation statistics files will be stored.
            start_step: the first step that runs subscriber actions.
            end_step: the end step (exclusively) that runs subscriber actions.
            override_output_dir: whether `output_dir` can be overridden if it already exists.
            run_on_cpu: whether to run the subscriber actions on CPU, this should be the last resort when inserted
                inspector node affects memory peak causing the original recipe run to fail with OOM.
            bucket_size: the size of the bucket to split the statistic calculation.
        """
        super().__init__(start_step=start_step, end_step=end_step)
        self._output_dir = output_dir
        self._run_on_cpu = run_on_cpu
        self._bucket_size = bucket_size

        if os.path.exists(self._output_dir):
            if override_output_dir:
                warnings.warn(f"Output directory {self._output_dir} already exists, overriding it.")
                shutil.rmtree(self._output_dir)
            else:
                raise FileExistsError(
                    f"Output directory {self._output_dir} already exists. "
                    "Set override_output_dir=True for StatisticsSubscriber if this is the intention."
                )

    def inspect_adhoc_activation(
        self, activation_name: str, tensor: torch.Tensor, run_ctx: _RuntimeStates
    ) -> torch.Tensor:
        return _ORTTensorInspector.apply(
            StatisticsSubscriber._post_forward_tensor_func_impl,
            StatisticsSubscriber._pre_backward_tensor_func_impl,
            activation_name,
            None,
            run_ctx,
            tensor,
            self._output_dir,
            self._run_on_cpu,
            self._bucket_size,
        )

    @staticmethod
    def _post_forward_tensor_func_impl(
        ctx,
        activation_name: str,
        module_idx: Optional[int],
        run_ctx: _RuntimeStates,
        input_tensor: torch.Tensor,
        output_dir: str,
        run_on_cpu: bool,
        bucket_size: int,
    ):
        """
        Args:
            ctx: context object to store intermediate information.
            activation_name: the name of the activation tensor.
            module_idx:
                For call case 1 - the unique id of the module that the activation belongs to, it is detected by the
                    SubscriberManager automatically.
                For call case 2 - e.g, _ORTTensorInspector is called by users (NOT by SubscriberManager), module_idx can
                    be None.
            run_ctx: runtime context.
                For call case 2 - need retrieve the runtime state from GlobalSubscriberManager.
            input_tensor: the activation tensor.

        Make sure there is a same number of `tensor` type inputs and outputs.
        This is enforced by ORT's PythonOp's schema check.
        """
        depth = -1
        if module_idx is not None:
            depth = run_ctx.global_states.module_index_to_depth[module_idx]

        input_tensor_copied = None
        if input_tensor is None or not isinstance(input_tensor, torch.Tensor):
            input_tensor_copied = input_tensor
        else:
            input_tensor_copied = input_tensor.detach().clone()

        ctx.current_step = run_ctx.global_states.execution_step
        ctx.name = activation_name
        ctx.id = module_idx
        ctx.depth = depth
        ctx.subscribers = run_ctx.global_states.subscribers
        ctx.output_dir = output_dir
        ctx.run_on_cpu = run_on_cpu
        ctx.bucket_size = bucket_size

        output_file_path = os.path.join(f"{output_dir}", f"step_{ctx.current_step}")
        StatisticsSubscriber._summarize_activations(
            input_tensor_copied, depth, activation_name, output_file_path, True, run_on_cpu, bucket_size
        )

        # # Run subscribers sequentially.
        # for subscriber in run_ctx.global_states.subscribers:
        #     subscriber.module_post_forward(input_tensor_copied, depth, activation_name, ctx.current_step)

        return input_tensor.detach() if input_tensor is not None else None

    @staticmethod
    def _pre_backward_tensor_func_impl(ctx, grad_output: torch.Tensor):
        val = None
        if grad_output is None or not isinstance(grad_output, torch.Tensor):
            val = grad_output
        else:
            val = grad_output.detach().clone()

        output_file_path = os.path.join(f"{ctx.output_dir}", f"step_{ctx.current_step}")

        StatisticsSubscriber._summarize_activations(
            val, ctx.depth, ctx.name, output_file_path, False, ctx.run_on_cpu, ctx.bucket_size
        )

        # for subscriber in ctx.subscribers:
        #     subscriber.module_pre_backward(val, ctx.depth, ctx.name, ctx.current_step)

        return None, None, None, None, None, grad_output.detach() if grad_output is not None else None, None, None, None

    def post_forward_tensor_func(
        self, activation_name: str, module_idx: Optional[int], run_ctx: _RuntimeStates, tensor: torch.Tensor
    ) -> torch.Tensor:
        if self._start_step <= run_ctx.global_states.execution_step < self._end_step:
            return _ORTTensorInspector.apply(
                StatisticsSubscriber._post_forward_tensor_func_impl,
                StatisticsSubscriber._pre_backward_tensor_func_impl,
                activation_name,
                module_idx,
                run_ctx,
                tensor,
                self._output_dir,
                self._run_on_cpu,
                self._bucket_size,
            )
        return tensor

    @staticmethod
    def _summarize_activations(
        tensor: torch.Tensor,
        depth: int,
        name: str,
        step_folder: str,
        is_forward: bool,
        run_on_cpu: bool,
        bucket_size: int,
    ):
        display_name = name + " forward run" if is_forward is True else name + " backward run"
        output_file_name = name + "_forward" if is_forward is True else name + "_backward"

        if tensor is None or not isinstance(tensor, torch.Tensor):
            print(f"{display_name} not a torch tensor, value: {tensor}")
            return

        step_path = Path(step_folder)
        if not step_path.exists():
            step_path.mkdir(parents=True, exist_ok=False)
        order_file_path = step_path / "order.txt"
        tensor_file_path = step_path / output_file_name

        # This is to try the best effort to align the count of numbers per line for easier comparison in diff views,
        # though it does not always guarantee to do this way.
        torch.set_printoptions(precision=6, linewidth=128)

        tensor_shape = tensor.shape
        tensor_dtype = tensor.dtype
        flatten_array = tensor.flatten().view(-1)

        if run_on_cpu:
            flatten_array = flatten_array.to("cpu")

        if run_on_cpu:
            num_nan = torch.isnan(flatten_array).sum()
            num_inf = torch.isinf(flatten_array).sum()
            num_neg = (flatten_array < 0).sum()
            num_pos = (flatten_array > 0).sum()
            num_zero = (flatten_array == 0).sum()
            min_value = flatten_array.min()
            max_value = flatten_array.max()
            mean_value = flatten_array.mean()
            std_value = flatten_array.std()
        else:
            # Split the calculation for each bucket, then do another round of calculation on the bucket results.
            # This can at the best effort reduce the peak memory impact.
            bucket_size = bucket_size
            element_count = flatten_array.numel()
            ceil_bucket_count = (element_count + bucket_size - 1) // (bucket_size)
            nan_buckets = torch.zeros(ceil_bucket_count, dtype=torch.int64, device=flatten_array.device)
            inf_buckets = torch.zeros(ceil_bucket_count, dtype=torch.int64, device=flatten_array.device)
            neg_buckets = torch.zeros(ceil_bucket_count, dtype=torch.int64, device=flatten_array.device)
            pos_buckets = torch.zeros(ceil_bucket_count, dtype=torch.int64, device=flatten_array.device)
            zero_buckets = torch.zeros(ceil_bucket_count, dtype=torch.int64, device=flatten_array.device)
            min_buckets = torch.zeros(ceil_bucket_count, dtype=flatten_array.dtype, device=flatten_array.device)
            max_buckets = torch.zeros(ceil_bucket_count, dtype=flatten_array.dtype, device=flatten_array.device)
            mean_buckets = torch.zeros(ceil_bucket_count, dtype=flatten_array.dtype, device=flatten_array.device)
            std_buckets = torch.zeros(ceil_bucket_count, dtype=flatten_array.dtype, device=flatten_array.device)

            # Summary for each bucket
            element_count_per_bucket = torch.zeros(ceil_bucket_count, dtype=torch.int64, device=flatten_array.device)
            for i in range(ceil_bucket_count):
                end = min((i + 1) * bucket_size, element_count)
                bucket = flatten_array[i * bucket_size : end]
                element_count_per_bucket[i] = bucket.numel()

                nan_buckets[i] = torch.isnan(bucket).sum()
                inf_buckets[i] = torch.isinf(bucket).sum()
                neg_buckets[i] = (bucket < 0).sum()
                pos_buckets[i] = (bucket > 0).sum()
                zero_buckets[i] = (bucket == 0).sum()
                min_buckets[i] = bucket.min()
                max_buckets[i] = bucket.max()
                mean_buckets[i] = bucket.sum()
                std_buckets[i] = bucket.std()

            # Reduction across all buckets
            num_nan = nan_buckets.sum()
            num_inf = inf_buckets.sum()
            num_neg = neg_buckets.sum()
            num_pos = pos_buckets.sum()
            num_zero = zero_buckets.sum()
            min_value = min_buckets.min()
            max_value = max_buckets.max()
            mean_value = float(mean_buckets.sum()) / float(element_count)
            # Here we refer to
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            # to calculate the combined standard deviation of all buckets.
            s = (element_count_per_bucket - 1) * (std_buckets**2) + element_count_per_bucket * (
                (mean_buckets - mean_value) ** 2
            )
            std_value = torch.sqrt(s.sum() / (element_count - 1))

        with order_file_path.open(mode="a", encoding="utf-8") as f:
            f.write(f"{output_file_name}\n")

        with tensor_file_path.open(mode="w", encoding="utf-8") as f:
            f.write(
                f"{'>'*max(0, depth) + display_name} shape: {tensor_shape} dtype: {tensor_dtype} size: {flatten_array.size()} \n"
                f"min: {min_value} max: {max_value}, mean: {mean_value}, "
                f"std: {std_value} \n"
                f"nan: {num_nan}, inf: {num_inf}\n"
            )
            f.write(f"samples(top 128): {flatten_array[:128]}\n")
            f.write(f"neg: {num_neg}, pos: {num_pos}, zero: {num_zero},\n")
            f.write(f"{'='*16}\n")
