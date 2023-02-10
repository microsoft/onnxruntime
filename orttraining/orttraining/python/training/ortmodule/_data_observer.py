# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
from onnx import helper
from onnx import onnx_pb as onnx_proto
import warnings
from onnxruntime.training import ortmodule
import torch


class DataObserver(object):
    """Configurable data observer for ORTModule.
    """

    def __init__(self, log_steps=1):
        self._enabled = ortmodule._defined_from_envvar("ORTMODULE_ENABLE_DATA_OBSERVER", 0, warn=True) == 1
        self._embedding_graph_input_to_padding_idx_map = {}
        self._embedding_stats = []

        self._loss_label_graph_input_to_padding_idx_map = {}
        self._loss_label_stats = []

        self._last_step = -1
        self._current_step = -1
        self._log_steps = log_steps

    def initialize_embedding_padding_inspector(self, model, user_input_names):
        """Register embedding input padding inspector.

        This is used for collecting data/compute sparsity information for embedding layer.
        """
        if not self._enabled:
            return

        self._embedding_graph_input_to_padding_idx_map.clear()

        def _get_initializer(model, name):
            for tensor in model.graph.initializer:
                if tensor.name == name:
                    return tensor

            return None

        for node in model.graph.node:
            if not (
                node.domain == "org.pytorch.aten" and node.op_type in ["ATen"] and node.input[1] in user_input_names
            ):
                continue

            found = [attr for attr in node.attribute if attr.name == "operator"]
            if not found or helper.get_attribute_value(found[0]).decode() != "embedding":
                continue

            tensor = _get_initializer(model, node.input[2])

            if tensor is None or tensor.data_type not in [onnx_proto.TensorProto.INT32, onnx_proto.TensorProto.INT64]:
                continue

            value = onnx.numpy_helper.to_array(tensor)
            if value.ndim != 0:
                warnings.warn(
                    "Embedding padding_idx must be a scalar, but got a tensor of shape {}".format(value.shape)
                )
                continue

            padding_idx = value.item()
            # Negative padding_idx in ATen embedding means there is no padding.
            if padding_idx < 0:
                continue

            if node.input[1] not in self._embedding_graph_input_to_padding_idx_map:
                self._embedding_graph_input_to_padding_idx_map[node.input[1]] = set()

            self._embedding_graph_input_to_padding_idx_map[node.input[1]].add(padding_idx)

    def inspect_from_input_data(self, name, data):
        if not self._enabled:
            return

        self._current_step += 1

        self._inspect_embedding_input(name, data)

    def _inspect_embedding_input(self, name, data):
        self._embedding_stats.clear()
        if (
            len(self._embedding_graph_input_to_padding_idx_map) > 0
            and name in self._embedding_graph_input_to_padding_idx_map
            and isinstance(data, torch.Tensor)
        ):
            for padding_idx in self._embedding_graph_input_to_padding_idx_map[name]:
                valid_token = torch.count_nonzero(data - padding_idx)
                total_token = data.numel()
                self._embedding_stats.append(
                    [name, padding_idx, float(valid_token) / float(total_token) * 100, valid_token, total_token]
                )

        if self._current_step - self._last_step >= self._log_steps:
            self._last_step = self._current_step
            self._print_embedding_stats()

    def _print_embedding_stats(self):
        if len(self._embedding_stats) > 0:
            stat = ">>>Embedding padding density (e.g. valid tokens/total tokens in current batch):\n"
            stat += "\t| {:<15} | {:<10} | {:<10} | {:<15} | {:<15} |\n".format(
                "INPUT NAME", "PAD IDX", "DENSITY", "VALID TOKENS", "TOTAL TOKENS"
            )
            for input_name, padding_idx, density, valid_token, total_token in self._embedding_stats:
                stat += "\t| {:<15} | {:<10} | {:<9.2f}% | {:<15} | {:<15} |\n".format(
                    input_name, padding_idx, density, valid_token, total_token
                )
            stat += "<<<\n"
            print(stat)
