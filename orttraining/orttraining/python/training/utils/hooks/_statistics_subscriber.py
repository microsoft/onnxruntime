# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import shutil
import warnings
from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional, Tuple, Union

import onnx
import torch

from ._subscriber_base import RuntimeStates, SubscriberBase
from ._subscriber_manager import ORT_NO_INCREASE_GLOBAL_STEP


class _InspectActivation(torch.autograd.Function):
    """
    This class is used to run the subscriber's forward and backward functions.
    The function will be called by two kinds of callers:
        1. SubscriberManager calls it for each registered nn.Module.
        2. Users who want to inspect the activation tensor at any place of model definition code.
    """

    @staticmethod
    def forward(
        ctx,
        activation_name: str,
        module_idx: Optional[int],
        run_ctx: RuntimeStates,
        input_tensor: torch.Tensor,
        module_post_forward,
        module_pre_backward,
    ):
        """
        Args:
            ctx: context object to store intermediate information.
            activation_name: the name of the activation tensor.
            module_idx: unit id of the module (address of the module instance).
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
        ctx.module_pre_backward = module_pre_backward

        module_post_forward(input_tensor_copied, depth, activation_name, ctx.current_step)

        return input_tensor.detach() if input_tensor is not None else None

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        val = None
        if grad_output is None or not isinstance(grad_output, torch.Tensor):
            val = grad_output
        else:
            val = grad_output.detach().clone()

        ctx.module_pre_backward(val, ctx.depth, ctx.name, ctx.current_step)

        return (
            None,
            None,
            None,
            grad_output.detach() if grad_output is not None else None,
            None,
            None,
        )

    @staticmethod
    def infer_shape(
        node: onnx.NodeProto,
        tensor_input_shapes: List[Optional[List[Union[int, str]]]],
        tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
    ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
        return tensor_input_shapes, tensor_input_dtypes

    @staticmethod
    def alias_input(node_proto_str: str):
        fw_alias_map = [3]
        bw_alias_map = [-1] * 6
        bw_alias_map[3] = 0
        return fw_alias_map, bw_alias_map


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

    def post_forward_tensor_apply_impl(
        self, run_rtx: RuntimeStates, module: torch.nn.Module, tensor_index: int, tensor: torch.Tensor
    ) -> torch.Tensor:
        module_index = run_rtx.global_states.module_to_module_index[module]
        name = f"{module.__class__.__name__}_{module_index}_{tensor_index}th_output"
        return _InspectActivation.apply(
            name, module_index, run_rtx, tensor, self.module_post_forward_impl, self.module_pre_backward_impl
        )

    def module_post_forward_impl(self, activation: torch.Tensor, depth: int, name: str, step: int):
        output_file_path = os.path.join(f"{self._output_dir}", f"step_{step}")
        return self._summarize_activations(activation, depth, name, output_file_path, True)

    def module_pre_backward_impl(self, activation: torch.Tensor, depth: int, name: str, step: int):
        output_file_path = os.path.join(f"{self._output_dir}", f"step_{step}")
        return self._summarize_activations(activation, depth, name, output_file_path, False)

    def _summarize_activations(self, tensor: torch.Tensor, depth: int, name: str, step_folder: str, is_forward: bool):
        display_name = name + " forward run" if is_forward is True else name + " backward run"
        output_file_name = name + "_forward" if is_forward is True else name + "_backward"

        # Skip dump during model pre-export output schema preparison run and export run.
        if ORT_NO_INCREASE_GLOBAL_STEP[0] is False:
            if tensor is None or not isinstance(tensor, torch.Tensor):
                print(f"{display_name} not a torch tensor, value: {tensor}")
                return

            step_path = Path(step_folder)
            if not step_path.exists():
                step_path.mkdir(parents=True, exist_ok=False)
            order_file_path = step_path / "order.txt"
            tensor_file_path = step_path / output_file_name

            with order_file_path.open(mode="a", encoding="utf-8") as f:
                f.write(f"{output_file_name}\n")

            with tensor_file_path.open(mode="w", encoding="utf-8") as f:
                _summarize_tensor(display_name, tensor, f, depth, self._run_on_cpu, self._bucket_size)


def _summarize_tensor(
    display_name: str,
    tensor: torch.Tensor,
    f: TextIOWrapper,
    depth: int = 0,
    run_on_cpu: bool = False,
    bucket_size: int = 1024 * 1024 * 1024 // 2,
):
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

            # Only calculate std for float types, otherwise it will throw exception.
            if bucket.dtype in [torch.float16, torch.float32, torch.float64]:
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

    f.write(
        f"{'>'*max(0, depth) + display_name} shape: {tensor_shape} dtype: {tensor_dtype} size: {flatten_array.size()} \n"
        f"min: {min_value} max: {max_value}, mean: {mean_value}, "
        f"std: {std_value} \n"
        f"nan: {num_nan}, inf: {num_inf}\n"
    )
    f.write(f"samples(top 128): {flatten_array[:128]}\n")
    f.write(f"neg: {num_neg}, pos: {num_pos}, zero: {num_zero},\n")
    f.write(f"{'='*16}\n")
