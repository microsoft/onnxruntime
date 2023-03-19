# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import warnings
import torch

from ._subscriber_manager import _ModuleHookSubscriberBase


class StatisticsSubscriber(_ModuleHookSubscriberBase):
    """
    This subscriber is used to dump the activation's statistics to a file.

    In the forward pass, summarize the tensor's statistics and write to a file.
    In the backward pass, summarize the tensor's statistics and write into another file.

    Each activation will be summarized into 1 or 2 files, depending on whether it is used in the backward pass.
    So for each run step, there will be many files.

    `merge_activation_summary.py` can be used to merge the files into one file per training step.
    """

    def __init__(self, output_dir, start_step_inclusive=0, end_step_exclusive=1000000, override_output_dir=False):
        super().__init__(start_step=start_step_inclusive, end_step=end_step_exclusive)
        self._output_dir = output_dir
        if os.path.exists(self._output_dir):
            if override_output_dir:
                import shutil

                warnings.warn(f"Output directory {self._output_dir} already exists, override it.")
                shutil.rmtree(self._output_dir)
            else:
                raise RuntimeError(
                    f"Output directory {self._output_dir} already exists. "
                    "Set override_output_dir=True for StatisticsSubscriber if this is the intention."
                )

    def forward(self, activation, depth, name, step):
        output_file_path = os.path.join(f"{self._output_dir}", f"step_{step}")
        return self._summarize_activations(activation, depth, name, output_file_path, True)

    def backward(self, activation, depth, name, step):
        output_file_path = os.path.join(f"{self._output_dir}", f"step_{step}")
        return self._summarize_activations(activation, depth, name, output_file_path, False)

    def _summarize_activations(self, tensor, depth, name, step_folder, is_forward):
        display_name = name + " forward run" if is_forward is True else name + " backward run"
        output_file_name = name + "_forward" if is_forward is True else name + "_backward"

        if tensor is None or not isinstance(tensor, torch.Tensor):
            print(f"{display_name} not a torch tensor, value: {tensor}")
            return None

        d = f"{step_folder}"
        isExist = os.path.exists(d)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(d)

        path = os.path.join(d, f"{output_file_name}")

        torch.set_printoptions(precision=6, linewidth=128)
        flatten_array = tensor.flatten()
        zerotensor = torch.tensor(0, dtype=flatten_array.dtype, device=flatten_array.device)
        num_nan = torch.isnan(flatten_array).sum()
        num_inf = torch.isinf(flatten_array).sum()

        with open(os.path.join(d, "order.txt"), "a") as of:
            of.write(f"{output_file_name}\n")

        with open(path, "w") as f:
            f.write(
                f"{'>'*depth + display_name} shape: {tensor.shape} dtype: {tensor.dtype} size: {flatten_array.size()} \n"
                f"min: {flatten_array.min()} max: {flatten_array.max()}, mean: {flatten_array.mean()}, "
                f"std: {flatten_array.std()} \n"
                f"nan: {num_nan}, inf: {num_inf}\n"
            )
            f.write(f"samples(top 128): {flatten_array[:128]}\n")

            f.write(
                f"neg: {torch.less(flatten_array, zerotensor).to(torch.int64).sum()}, "
                f"pos: {torch.greater(flatten_array, zerotensor).to(torch.int64).sum()}, "
                f"zero: {torch.eq(flatten_array, zerotensor).to(torch.int64).sum()},\n"
            )
            f.write(f"{'='*16}\n")
