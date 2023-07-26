# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from enum import IntEnum
from logging import Logger
from typing import List, Tuple, Union

import onnx
import torch
from onnx import ModelProto, helper
from onnx import onnx_pb as onnx_proto


class Phase(IntEnum):
    INVALID = -1
    PRE_FORWARD = 0
    POST_FORWARD = 1
    PRE_BACKWARD = 2  # not applicable for inference
    POST_BACKWARD = 3  # not applicable for inference


def _convert_phase_to_string(phase: Phase) -> str:
    if phase == Phase.PRE_FORWARD:
        return "pre_forward"
    elif phase == Phase.POST_FORWARD:
        return "post_forward"
    elif phase == Phase.PRE_BACKWARD:
        return "pre_backward"
    elif phase == Phase.POST_BACKWARD:
        return "post_backward"
    else:
        return "invalid"


class RuntimeInspector:
    """
    Runtime inspector for ORTModule.
    """

    def __init__(self, logger: Logger):
        self._logger = logger

        self.input_density_ob: Union[InputDensityObserver, None] = None
        self.memory_ob: Union[MemoryObserver, None] = None

    def enable_input_inspector(self, model: ModelProto, user_input_names: List[str]) -> None:
        """Initialize input inspector from the given ONNX model and user input names.

        Args:
            model: ONNX model.
            user_input_names: User input names in the ONNX model.

        """
        if self.input_density_ob is None:
            self.input_density_ob = InputDensityObserver(self._logger)
        else:
            raise RuntimeError("Input density observer is already enabled.")

        return self.input_density_ob.initialize(model, user_input_names)

    def inspect_input(self, input_name, input_data) -> Tuple[bool, float, float]:
        """Inspect input data and print statistics.

        Args:
            input_name: User input name.
            input_data: User input tensor.

        Returns:
            found: Whether the input name is found in `_embedding_graph_input_to_padding_idx_map` and
                `_loss_label_graph_input_to_ignore_idx_map`.
            embed_input_density: Density for the inspected embedding input if found to be True; otherwise, return 100.
            label_input_density: Density for the inspected label input if found to be True; otherwise, return 100.
        """
        if self.input_density_ob is not None:
            return self.input_density_ob.inspect_from_input_data(input_name, input_data)

        return (False, 100, 100)

    def disable_input_inspector(self) -> None:
        """Disable input density inspector."""
        self.input_density_ob = None

    def enable_memory_inspector(self, module: torch.nn.Module):
        """Enable memory inspector for ORTModule.

        Args:
            module: ORTModule.
        """
        if self.memory_ob is None:
            self.memory_ob = MemoryObserver(module, self._logger)
        else:
            raise RuntimeError("Memory observer is already enabled.")

    def inspect_memory(self, phase: Phase) -> None:
        """Inspect memory usage and print statistics.

        Args:
            phase: Phase to inspect.
        """
        if self.memory_ob is not None:
            self.memory_ob.inspect_memory(phase)


class InputDensityObserver:
    """Training input data observer for ORTModule.

    Data observer is used to collect data/compute sparsity information for embedding and label inputs. It needs to be
    firstly initialized with the ONNX model and user input names. Then, it can be used to inspect the input data
    through `inspect_from_input_data()` method given user input name and input tensor. Inspection results will be
    printed per `log_steps`.

    """

    def __init__(self, logger: Logger, log_steps=1):
        self._logger = logger
        self._embedding_graph_input_to_padding_idx_map = {}
        self._loss_label_graph_input_to_ignore_idx_map = {}
        self._stats = []
        self._is_initialized = False

        self._last_step = 0
        self._current_step = 0
        self._log_steps = log_steps

        self._tensor_to_node_map = {}

    def initialize(self, model: ModelProto, user_input_names: List[str]) -> None:
        """Initialize data observer from the given ONNX model and user input names.

        For embedding input (e.g. ATen embedding), try to parse the padding_idx from the ONNX model, if padding_idx is
        valid, register it in _embedding_graph_input_to_padding_idx_map.
        For label input (e.g. SoftmaxCrossEntropyLossInternal), try to parse the ignore_index from the ONNX model, if
        ignore_index is valid, register it in _loss_label_graph_input_to_ignore_idx_map.

        Args:
            model: ONNX model.
            user_input_names: User input names in the ONNX model.

        """
        if self._is_initialized:
            return

        try:
            self._tensor_to_node_map.clear()
            for node in model.graph.node:
                for output_name in node.output:
                    if output_name != "":
                        self._tensor_to_node_map[output_name] = node

            self._initialize_embedding_padding_inspector(model, user_input_names)
            self._initialize_loss_label_padding_inspector(model, user_input_names)

            self._is_initialized = True
        except Exception as e:
            self._is_initialized = False
            self._logger.warning(f"Failed to initialize InputDensityObserver due to {e}")

    def _initialize_embedding_padding_inspector(self, model, user_input_names):
        """Register embedding input padding inspector.

        Iterate all ATen embedding nodes, and check if the following conditions are met:
        > 1. the data input is ONNX graph input;
        > 2. and the padding_idx is a non-negative scalar constant.

        If yes, append <ONNX graph input, padding_idx> into _embedding_graph_input_to_padding_idx_map, which is later
        used for collecting data/compute sparsity information for embedding layer.
        """

        self._embedding_graph_input_to_padding_idx_map.clear()

        for node in model.graph.node:
            if not (
                node.domain == "org.pytorch.aten"
                and node.op_type == "ATen"
                and node.input[1] in user_input_names
                and len(node.input) >= 3
            ):
                continue

            found = [attr for attr in node.attribute if attr.name == "operator"]
            if not found or helper.get_attribute_value(found[0]).decode() != "embedding":
                continue

            tensor = None
            padding_const_node = self._try_get_node_from_its_output(node.input[2])
            if padding_const_node is None:
                padding_initializer_name = node.input[2]
                tensor = self._try_get_initializer(model, padding_initializer_name)

            elif padding_const_node.op_type == "Constant":
                found = [attr for attr in padding_const_node.attribute if attr.name == "value"]
                tensor = found[0].t
            else:
                continue

            if tensor is None or tensor.data_type not in [onnx_proto.TensorProto.INT32, onnx_proto.TensorProto.INT64]:
                continue

            value = onnx.numpy_helper.to_array(tensor)
            if value.ndim != 0:
                self._logger.warning(f"Embedding padding_idx must be a scalar, but got a tensor of shape {value.shape}")
                continue

            padding_idx = value.item()
            # Negative padding_idx in ATen embedding means there is no padding.
            if padding_idx < 0:
                continue

            if node.input[1] not in self._embedding_graph_input_to_padding_idx_map:
                self._embedding_graph_input_to_padding_idx_map[node.input[1]] = set()

            self._embedding_graph_input_to_padding_idx_map[node.input[1]].add(padding_idx)

    def _initialize_loss_label_padding_inspector(self, model, user_input_names):
        """Register loss label input padding inspector.

        Iterate all SoftmaxCrossEntropyLossInternal nodes, and check if the following conditions are met:
        > 1. ignore_index (the 4th input) is a non-negative scalar constant;
        > 2. label input (the 2nd input) is either a). ONNX graph input or b). a Reshape node with a Slice node as its
          input. In the case of b), the Slice node must be a pattern defined in Bloom model.

        If yes, append <ONNX graph input, <ignore_index, label_preprocess_function>> into
        _loss_label_graph_input_to_ignore_idx_map, which is later used for collecting data/compute sparsity information
        for labels.
        """

        def _default_label_preprocess(labels):
            return labels

        self._loss_label_graph_input_to_ignore_idx_map.clear()
        for node in model.graph.node:
            if not (
                node.domain == "com.microsoft"
                and node.op_type == "SoftmaxCrossEntropyLossInternal"
                and len(node.input) == 4
            ):
                continue

            tensor = None
            padding_const_node = self._try_get_node_from_its_output(node.input[3])
            if padding_const_node is None:
                padding_initializer_name = node.input[3]
                tensor = self._try_get_initializer(model, padding_initializer_name)

            elif padding_const_node.op_type == "Constant":
                found = [attr for attr in padding_const_node.attribute if attr.name == "value"]
                tensor = found[0].t
            else:
                continue

            if tensor is None or tensor.data_type not in [onnx_proto.TensorProto.INT32, onnx_proto.TensorProto.INT64]:
                continue

            value = onnx.numpy_helper.to_array(tensor)
            if value.ndim != 0:
                self._logger.warning(
                    f"SoftmaxCrossEntropyLossInternal ignore_index must be a scalar, but got a tensor of shape {value.shape}"
                )
                continue

            ignore_index = value.item()

            # Check label inputs
            label_graph_input = None

            label_preprocess_func = _default_label_preprocess
            reshape_node = self._try_get_node_from_its_output(node.input[1])
            # The label input comes from graph input or a Reshape node consuming a graph input, which is aligned with
            # orttraining/orttraining/core/optimizer/compute_optimizer/sceloss_compute_optimization.cc.
            if reshape_node is None:
                if node.input[1] not in user_input_names:
                    continue
                label_graph_input = node.input[1]
            else:
                if reshape_node.op_type != "Reshape":
                    continue

                reshape_input = reshape_node.input[0]
                if reshape_input in user_input_names:
                    label_graph_input = reshape_input
                else:  # Pattern defined in Bloom model.
                    slice_node = self._try_get_node_from_its_output(reshape_input)
                    if slice_node is None:
                        continue

                    if slice_node.op_type != "Slice":
                        continue

                    slice_input = slice_node.input[0]
                    starts = self._try_get_initializer_value(model, slice_node.input[1])
                    ends = self._try_get_initializer_value(model, slice_node.input[2])
                    axes = self._try_get_initializer_value(model, slice_node.input[3])
                    steps = self._try_get_initializer_value(model, slice_node.input[4])
                    if (
                        slice_input in user_input_names
                        and starts is not None
                        and ends is not None
                        and axes is not None
                        and steps is not None
                        and len(axes) == 1
                        and axes[0] == 1
                    ):
                        label_graph_input = slice_input

                        def _slice_label_preprocess(labels, s=starts[0], e=ends[0], st=steps[0]):
                            return labels[:, s:e:st]

                        label_preprocess_func = _slice_label_preprocess

                if label_graph_input is None:
                    continue

            if label_graph_input not in self._loss_label_graph_input_to_ignore_idx_map:
                self._loss_label_graph_input_to_ignore_idx_map[label_graph_input] = []

            self._loss_label_graph_input_to_ignore_idx_map[label_graph_input].append(
                [ignore_index, label_preprocess_func]
            )

    def inspect_from_input_data(self, name: str, inp) -> Tuple[bool, float, float]:
        """Inspect input data and print statistics.

        Args:
            name: User input name.
            inp: User input tensor.
        Returns:
            found: Whether the input name is found in `_embedding_graph_input_to_padding_idx_map` and
                `_loss_label_graph_input_to_ignore_idx_map`.
            embed_input_density: Density for the inspected embedding input if found to be True; otherwise, return 100.
            label_input_density: Density for the inspected label input if found to be True; otherwise, return 100.
        """
        if not self._is_initialized:
            return (False, 100, 100)

        try:
            data = inp.clone()
            found, embed_input_density, label_input_density = self._inspect_embed_label_input(name, data)
            if found:
                self._current_step += 1

                if self._current_step - self._last_step >= self._log_steps:
                    self._last_step = self._current_step
                    self._print_embed_label_stats()

            return (found, embed_input_density, label_input_density)
        except Exception as e:
            self._logger.warning(f"Failed to inspect input {name} due to {e}", UserWarning)
            return (False, 100, 100)

    def _inspect_embed_label_input(self, name, data):
        found = False
        min_embed_density = 100
        min_label_density = 100
        if (
            len(self._embedding_graph_input_to_padding_idx_map) > 0
            and name in self._embedding_graph_input_to_padding_idx_map
            and isinstance(data, torch.Tensor)
        ):
            for padding_idx in self._embedding_graph_input_to_padding_idx_map[name]:
                valid_token = torch.count_nonzero(data - padding_idx)
                valid_token_per_batch = "N/A"
                if data.dim() > 1:
                    valid_token_per_batch = str(torch.count_nonzero(data - padding_idx, dim=1).tolist())
                total_token = data.numel()
                embed_density = float(valid_token) / float(total_token) * 100
                if embed_density < 90:
                    min_embed_density = min(min_embed_density, embed_density)
                self._stats.append(
                    [
                        self._current_step,
                        "EMBED",
                        name,
                        padding_idx,
                        embed_density,
                        valid_token,
                        total_token,
                        valid_token_per_batch,
                    ]
                )
                found = True

        if (
            len(self._loss_label_graph_input_to_ignore_idx_map) > 0
            and name in self._loss_label_graph_input_to_ignore_idx_map
            and isinstance(data, torch.Tensor)
        ):
            for ignore_index, preprocess_func in self._loss_label_graph_input_to_ignore_idx_map[name]:
                data_preprocessed = preprocess_func(data)
                valid_token = torch.count_nonzero(data_preprocessed - ignore_index)
                total_token = data_preprocessed.numel()
                label_density = float(valid_token) / float(total_token) * 100
                if label_density < 90:
                    min_label_density = min(min_label_density, label_density)
                self._stats.append(
                    [
                        self._current_step,
                        "LABEL",
                        name,
                        ignore_index,
                        label_density,
                        valid_token,
                        total_token,
                        "N/A",
                    ]
                )
                found = True

        return found, min_embed_density, min_label_density

    def _print_embed_label_stats(self):
        if len(self._stats) > 0:
            stat = f">>>Valid token/label density (e.g. valid/total) in passing {self._log_steps} steps:\n"
            stat += "\t| {:<10} | {:<10} | {:<15} | {:<10} | {:<10} | {:<15} | {:<15} | {:<15} |\n".format(
                "STEP",
                "INPUT TYPE",
                "INPUT NAME",
                "PAD IDX",
                "DENSITY",
                "VALID TOKENS",
                "TOTAL TOKENS",
                "VALID TOKENS/BATCH",
            )
            for (
                step,
                input_type,
                input_name,
                padding_idx,
                density,
                valid_token,
                total_token,
                valid_token_per_batch,
            ) in self._stats:
                stat += "\t| {:<10} | {:<10} | {:<15} | {:<10} | {:<9.2f}% | {:<15} | {:<15} | {:<15} |\n".format(
                    step, input_type, input_name, padding_idx, density, valid_token, total_token, valid_token_per_batch
                )
            stat += "<<<\n"
            self._logger.info(stat)
            self._stats.clear()

    def _try_get_node_from_its_output(self, name):
        if name == "" or name not in self._tensor_to_node_map:
            return None

        return self._tensor_to_node_map[name]

    def _try_get_initializer(self, model, name):
        for tensor in model.graph.initializer:
            if tensor.name == name:
                return tensor

        return None

    def _try_get_initializer_value(self, model, name):
        tensor = self._try_get_initializer(model, name)
        if tensor is None:
            return None
        value = onnx.numpy_helper.to_array(tensor)
        return value


class MemoryObserver:
    """Memory inspector across the training lifetime.

    On different training/inference phases, `inspect_memory` is called to print out the memory usage, including
    current/peak memory usage, current/peak inactive and non-releasable memory.
    """

    NORMALIZER_FACTOR = float(1024 * 1024)
    NORMALIZER_UNIT = "MiB"

    def __init__(self, m: torch.nn.Module, logger: Logger):
        self._logger = logger
        self._current_step = 0
        self._rank = 0
        self._world_size = 1
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

        self._rank_info = f"[{self._rank}/{self._world_size}]"
        self._pre_phase = Phase.INVALID
        self._last_phase = Phase.POST_BACKWARD if m.training else Phase.POST_FORWARD

        self._is_first_inspect = True

    def inspect_memory(self, cur_phase: Phase):
        if not torch.cuda.is_available():
            return

        if self._is_first_inspect:
            # Clean the memory cache and memory stats before the first time run forward pass, FOR EVERY RANK.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self._is_first_inspect = False

        if self._rank != 0:
            return

        if cur_phase < Phase.PRE_FORWARD or cur_phase > self._last_phase:
            raise RuntimeError(f"Invalid phase detected: {cur_phase}")

        if (cur_phase - self._pre_phase) != 1:
            raise RuntimeError(f"Invalid phase transition detected: {self._pre_phase} -> {cur_phase}")

        cur_mem_allocated = self._normalize(torch.cuda.memory_allocated())
        max_mem_allocated = self._normalize(torch.cuda.max_memory_allocated())
        cur_mem_cached = self._normalize(torch.cuda.memory_reserved())
        max_mem_cached = self._normalize(torch.cuda.max_memory_reserved())
        torch_mem_stat = torch.cuda.memory_stats()
        cur_mem_inactive = self._normalize(torch_mem_stat.get("inactive_split_bytes.all.current", 0))
        max_mem_inactive = self._normalize(torch_mem_stat.get("inactive_split_bytes.all.peak", 0))

        mem_stats = [
            ["phase", _convert_phase_to_string(cur_phase)],
            ["allocated", cur_mem_allocated],  # current memory alloeated for tensors
            ["max allocated", max_mem_allocated],  # peak memory allocated for tensors
            ["cached", cur_mem_cached],  # current memory cached for caching allocator
            ["max cached", max_mem_cached],  # peak memory cached for caching allocator.
            ["inactive", cur_mem_inactive],  # amount of inactive, non-releasable memory
            ["max inactive", max_mem_inactive],  # peak of inactive, non-releasable memory
        ]

        summ = f"{self._rank_info} step {self._current_step} memory ({MemoryObserver.NORMALIZER_UNIT})"
        for stat in mem_stats:
            summ += f" | {stat[0]}: {stat[1]}"

        # For the 10+ steps, only print when it is power of 2.
        if self._current_step < 10 or (self._current_step & (self._current_step - 1) == 0):
            self._logger.info(summ)

        if cur_phase == self._last_phase:
            self._increase_step()
            self._pre_phase = Phase.INVALID
            return

        self._pre_phase = cur_phase

    def _increase_step(self):
        self._current_step += 1

    def _normalize(self, mem_size_in_bytes: Union[float, int]) -> str:
        return f"{float(mem_size_in_bytes) / MemoryObserver.NORMALIZER_FACTOR:.0f}"
