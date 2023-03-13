# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import warnings
import onnx
import os
import torch

from collections import abc
from onnx import helper
from onnx import onnx_pb as onnx_proto
from onnxruntime.training import ortmodule


class RuntimeInspector:
    def __init__(self):
        self.input_density_ob = InputDensityObserver()


class InputDensityObserver:
    """Configurable input data observer for ORTModule."""

    def __init__(self, log_steps=10):
        self._enabled = ortmodule._defined_from_envvar("ORTMODULE_ENABLE_INPUT_DENSITY_INSPECTOR", 0, warn=True) == 1
        self._embedding_graph_input_to_padding_idx_map = {}
        self._loss_label_graph_input_to_ignore_idx_map = {}
        self._stats = []
        self._is_initialized = False

        self._last_step = 0
        self._current_step = 0
        self._log_steps = log_steps

        self._tensor_to_node_map = {}

    def initialize(self, model, user_input_names):
        """Initialize data observer."""
        if not self._enabled:
            return

        self._tensor_to_node_map.clear()
        for node in model.graph.node:
            for output_name in node.output:
                if output_name != "":
                    self._tensor_to_node_map[output_name] = node

        self._initialize_embedding_padding_inspector(model, user_input_names)
        self._initialize_loss_label_padding_inspector(model, user_input_names)

        self._is_initialized = True

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
            if not (node.domain == "org.pytorch.aten" and node.op_type == "ATen" and node.input[1] in user_input_names):
                continue

            found = [attr for attr in node.attribute if attr.name == "operator"]
            if not found or helper.get_attribute_value(found[0]).decode() != "embedding":
                continue

            tensor = self._try_get_initializer(model, node.input[2])
            if tensor is None or tensor.data_type not in [onnx_proto.TensorProto.INT32, onnx_proto.TensorProto.INT64]:
                continue

            value = onnx.numpy_helper.to_array(tensor)
            if value.ndim != 0:
                warnings.warn(f"Embedding padding_idx must be a scalar, but got a tensor of shape {value.shape}")
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
        if not self._enabled:
            return

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

            tensor = self._try_get_initializer(model, node.input[3])
            if tensor is None or tensor.data_type not in [onnx_proto.TensorProto.INT32, onnx_proto.TensorProto.INT64]:
                continue

            value = onnx.numpy_helper.to_array(tensor)
            if value.ndim != 0:
                warnings.warn(
                    f"SoftmaxCrossEntropyLossInternal ignore_index must be a scalar, but got a tensor of shape {value.shape}"
                )
                continue

            ignore_index = value.item()

            # Check label inputs
            label_graph_input = None

            label_preprocess_func = _default_label_preprocess
            reshape_node = self._try_get_node_from_its_output(node.input[1])
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

    def inspect_from_input_data(self, name, inp):
        if not self._enabled or not self._is_initialized:
            return

        data = inp.clone()
        found = self._inspect_embed_label_input(name, data)
        if found:
            self._current_step += 1

            if self._current_step - self._last_step >= self._log_steps:
                self._last_step = self._current_step
                self._print_embed_label_stats()

    def _inspect_embed_label_input(self, name, data):
        found = False
        if (
            len(self._embedding_graph_input_to_padding_idx_map) > 0
            and name in self._embedding_graph_input_to_padding_idx_map
            and isinstance(data, torch.Tensor)
        ):
            for padding_idx in self._embedding_graph_input_to_padding_idx_map[name]:
                valid_token = torch.count_nonzero(data - padding_idx)
                valid_token_per_batch = torch.count_nonzero(data - padding_idx, dim=1)
                total_token = data.numel()
                self._stats.append(
                    [
                        self._current_step,
                        "EMBED",
                        name,
                        padding_idx,
                        float(valid_token) / float(total_token) * 100,
                        valid_token,
                        total_token,
                        str(valid_token_per_batch.tolist()),
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
                self._stats.append(
                    [
                        self._current_step,
                        "LABEL",
                        name,
                        ignore_index,
                        float(valid_token) / float(total_token) * 100,
                        valid_token,
                        total_token,
                        "N/A",
                    ]
                )
                found = True

        return found

    def _print_embed_label_stats(self):
        if len(self._stats) > 0:
            stat = f">>>Valid token/label density (e.g. valid/total) in passing {self._log_steps} steps:\n"
            stat += "\t| {:<10} | {:<10} | {:<15} | {:<10} | {:<10} | {:<15} | {:<15} | {:<15} |\n".format(
                "STEP",
                "INPUT TYPE",
                " INPUT NAME",
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
            print(stat)
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


def summarize_activations(tensor, depth, name, step_folder):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        print(f"{name} not a torch tensor, value: {tensor}")
        return None

    d = f"{step_folder}"
    isExist = os.path.exists(d)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(d)

    path = os.path.join(d, f"{name}")

    torch.set_printoptions(precision=6, linewidth=128)
    flatten_array = tensor.flatten()
    zerotensor = torch.tensor(0, dtype=flatten_array.dtype, device=flatten_array.device)
    num_nan = torch.isnan(flatten_array).sum()
    num_inf = torch.isinf(flatten_array).sum()

    with open(os.path.join(d, "order.txt"), "a") as of:
        of.write(f"{name}\n")

    f = open(path, "w")
    f.write(
        f"{'>'*depth + name} shape: {tensor.shape} dtype: {tensor.dtype} size: {flatten_array.size()} \n"
        f"min: {flatten_array.min()} max: {flatten_array.max()}, mean: {flatten_array.mean()}, std: {flatten_array.std()} \n"
        f"nan: {num_nan}, inf: {num_inf}\n"
    )
    f.write(f"samples(top 128): {flatten_array[:128]}\n")

    f.write(
        f"neg: {torch.less(flatten_array, zerotensor).to(torch.int64).sum()}, pos: {torch.greater(flatten_array, zerotensor).to(torch.int64).sum()}, zero: {torch.eq(flatten_array, zerotensor).to(torch.int64).sum()},\n"
    )
    # if num_nan + num_inf == 0:
    #     elem_count = int(flatten_array.size(dim=0))
    #     f.write(f"histogram: {torch.histogram(flatten_array, bins=max(1, elem_count.bit_length() - 1)) if elem_count > 2 else None} \n")
    f.write(f"==================================================================\n")
    f.close()


DUMP_GLOBAL_STEP = 0  # Used to track dump step
CONSTANT_OUTPUT_FOLDER_NAME = None  # Used to specify the folder name to dump
CONSTANT_START_STEP = None
CONSTANT_END_STEP = None


# Used to store the dumped activation names, if already dumped, then skipped.
# Need to clear after each step.
OBSERVED_ACTIVATION_NAME = {}
MODULE_INDEX_TO_DEPTH = {}  # Used to store the depth of each module
MODULE_TO_MODULE_INDEX = {}  # Used to store the index of each module


def is_builtin_type(obj):
    # https://stackoverflow.com/a/17795199
    return obj.__class__.__module__ == "__builtin__" or obj.__class__.__module__ == "builtins"


def _apply_to_tensors(module, id, tensors, func):
    idx = [0]

    def flatten_outputs(outputs, id, index, func):
        if isinstance(outputs, abc.Sequence):
            touched_outputs = []
            for output in outputs:
                touched_output = _apply_to_tensors(module, id, output, func)
                touched_outputs.append(touched_output)
            return outputs.__class__(touched_outputs)
        elif isinstance(outputs, abc.Mapping):
            # apply inplace to avoid recreating dict inherited objects
            for key, value in outputs.items():
                outputs[key] = _apply_to_tensors(module, id, outputs[key], func)
            return outputs

        elif type(outputs) is torch.Tensor:
            cur_id = index[0]
            index[0] += 1
            return func(cur_id, outputs)

        else:
            if not is_builtin_type(outputs):
                raise RuntimeError(f"Unknown type {type(outputs)}")
            return outputs

    return flatten_outputs(tensors, id, idx, func)


class ActivationComplete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        global DUMP_GLOBAL_STEP
        ctx.current_step = DUMP_GLOBAL_STEP
        return input

    @staticmethod
    def backward(ctx, grad_output):
        global DUMP_GLOBAL_STEP
        global OBSERVED_ACTIVATION_NAME

        # This backward is called for each gradient input of the module, so we only increase the dump step
        # and clear states once.
        if ctx.current_step == DUMP_GLOBAL_STEP:
            print(f"================== Completed forward pass dump for STEP {ctx.current_step} ==================")
            DUMP_GLOBAL_STEP += 1
            OBSERVED_ACTIVATION_NAME = {}

        return grad_output


class ActivationObserver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, id, depth, input):
        global DUMP_GLOBAL_STEP
        global CONSTANT_OUTPUT_FOLDER_NAME
        val = None
        if input is None or not isinstance(input, torch.Tensor):
            val = input
        else:
            val = input.detach().clone()

        ctx.current_step = DUMP_GLOBAL_STEP

        global CONSTANT_START_STEP
        global CONSTANT_END_STEP

        if CONSTANT_START_STEP is not None and CONSTANT_END_STEP is not None:
            if ctx.current_step >= CONSTANT_START_STEP and ctx.current_step < CONSTANT_END_STEP:
                summarize_activations(
                    val, depth, name + " forward run", os.path.join(f"{CONSTANT_OUTPUT_FOLDER_NAME}", f"step_{ctx.current_step}")
                )
        ctx.name = name
        ctx.id = id
        ctx.depth = depth

        return input.detach() if input is not None else None

    @staticmethod
    def backward(ctx, grad_output):
        global OBSERVED_ACTIVATION_NAME
        val = None

        if grad_output is None or not isinstance(grad_output, torch.Tensor):
            val = grad_output
        else:
            val = grad_output.detach().clone()

        if CONSTANT_START_STEP is not None and CONSTANT_END_STEP is not None:
            if ctx.current_step >= CONSTANT_START_STEP and ctx.current_step < CONSTANT_END_STEP:
                summarize_activations(
                    val, ctx.depth, ctx.name + " backward run", os.path.join(f"{CONSTANT_OUTPUT_FOLDER_NAME}", f"step_{ctx.current_step}")
                )

        return None, None, None, grad_output.detach() if grad_output is not None else None


class ActivationSummarizer:
    def __init__(self, name, start_step_inclusive=0, end_step_exclusive=1000000):
        global CONSTANT_OUTPUT_FOLDER_NAME
        global CONSTANT_START_STEP
        global CONSTANT_END_STEP
        CONSTANT_OUTPUT_FOLDER_NAME = name
        CONSTANT_START_STEP = start_step_inclusive
        CONSTANT_END_STEP = end_step_exclusive

    def initialize(self, module):
        count = [0]
        # Register post forward hook for every module, inside the hook, we loop every tensor output of the module,
        # and wrap it with a autograd Function called ActivationObserver (which take in a tensor and return the same
        # tensor). In this way, we keep ORT and PyTorch run have the same boundary to check activation equality.
        self._register_hooks_recursively(module, 1, count)

        # Register backward start hook, then we increase the dump step.
        # Be noted, if backward is not triggered, the global dump step remain the original number,
        # which means the subsequent run will override the previous dump files. This indeed happens imagine ORTModule
        # firstly export graph (run the forward only), after gradient graph is built, another forward+backward is
        # triggered, override the previous dump files.
        def _post_backward_module_hook(module, inputs, outputs2):
            def _apply_to_tensors_func(index, outputs):
                return ActivationComplete.apply(outputs)

            return _apply_to_tensors(module, 0, outputs2, _apply_to_tensors_func)

        module.register_forward_hook(_post_backward_module_hook)

    def _register_hooks_recursively(self, module, depth, count):
        id = count[0]
        global MODULE_TO_MODULE_INDEX
        global MODULE_INDEX_TO_DEPTH
        MODULE_INDEX_TO_DEPTH[id] = depth
        MODULE_TO_MODULE_INDEX[module] = id

        for child in module.children():
            if isinstance(child, torch.nn.Module) and child not in MODULE_TO_MODULE_INDEX:
                count[0] += 1
                self._register_hooks_recursively(child, depth + 1, count)

        module.register_forward_hook(ActivationSummarizer._post_forward_module_hook)

    @staticmethod
    def _post_forward_module_hook(module, input, output):
        global MODULE_TO_MODULE_INDEX
        global MODULE_INDEX_TO_DEPTH
        global OBSERVED_ACTIVATION_NAME

        if module not in MODULE_TO_MODULE_INDEX:
            return

        id = MODULE_TO_MODULE_INDEX[module]
        if not isinstance(module, torch.nn.Module):
            return

        def _apply_to_tensors_func(index, outputs):
            global OBSERVED_ACTIVATION_NAME
            name = f"{module.__class__.__name__}_{id}_{index}th_output"
            if name not in OBSERVED_ACTIVATION_NAME:
                OBSERVED_ACTIVATION_NAME[name] = True
                return ActivationObserver.apply(name, id, MODULE_INDEX_TO_DEPTH[id], outputs)
            else:
                return outputs

        return _apply_to_tensors(module, id, output, _apply_to_tensors_func)
