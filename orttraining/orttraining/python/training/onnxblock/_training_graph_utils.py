# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import logging
import os
from typing import List, Optional, Set, Tuple, Union

import onnx

from onnxruntime import SessionOptions
from onnxruntime.capi._pybind_state import GradientGraphBuilder, get_optimized_model


def _disable_training_mode(model: onnx.ModelProto) -> None:
    """Disables the training mode of the model by removing the training configuration."""

    def disable_training_mode_dropout(node):
        # Training mode is the third input of Dropout
        if len(node.input) > 2:
            node.input[2] = ""

    def disable_training_mode_batchnorm(node):
        # Training mode is an attribute of BatchNormalization
        for attr in node.attribute:
            if attr.name == "training_mode":
                attr.i = 0

    ops_to_disable_training_mode_func_map = {
        "Dropout": disable_training_mode_dropout,
        "BatchNormalization": disable_training_mode_batchnorm,
    }
    for node in model.graph.node:
        if node.op_type in ops_to_disable_training_mode_func_map:
            ops_to_disable_training_mode_func_map[node.op_type](node)


def _reorder_outputs(model: onnx.ModelProto, user_output_names: List[str], requires_grad: Set[str]) -> None:
    """Reorders the outputs of the model to match the order of [user_outputs, gradients]"""

    graph_outputs = {output.name: output for output in model.graph.output}
    ordered_graph_outputs = [graph_outputs[name] for name in user_output_names]

    for arg in model.graph.input:
        if arg.name in requires_grad:
            gradient_name = f"{arg.name}_grad"
            ordered_graph_outputs.append(graph_outputs[gradient_name])

    del model.graph.output[:]
    model.graph.output.extend(ordered_graph_outputs)


def _move_initializers_to_inputs(model: onnx.ModelProto, initializer_names: Optional[Set[str]] = None) -> None:
    # Move all trainable and non trainable initializers to graph inputs.
    # This allows training to pass in the parameters from outside the graph
    # so as to share the parameters across multiple sessions.
    initializers = []
    for initializer in model.graph.initializer:
        if not initializer_names or initializer.name in initializer_names:
            model.graph.input.append(
                onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)
            )
        else:
            initializers.append(initializer)

    # Update the initializers in the graph
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def _gradient_model_for(
    model: onnx.ModelProto,
    requires_grad: Set[str],
    loss_name: str,
    options: Optional[SessionOptions] = None,
) -> onnx.ModelProto:
    """Builds the gradient graph on top of the given input forward only graph."""

    logging.debug(
        "The loss output is %s. The gradient graph will be built starting from %s_grad.", loss_name, loss_name
    )

    builder = GradientGraphBuilder(model.SerializeToString(), {loss_name}, requires_grad, loss_name, options)
    builder.build()
    return onnx.load_from_string(builder.get_model())


def build_gradient_graph(
    model: onnx.ModelProto,
    requires_grad: Set[str],
    frozen_params: Set[str],
    output_names: Union[List[str], str],
    custom_op_library: Optional[str] = None,
) -> Tuple[onnx.ModelProto, onnx.ModelProto]:
    """Prepare the training model and the eval model.

    This function will restructure the model to prepare for training.
    - Move all initializers to graph inputs (so they can be shared between training/eval sessions)
    - Run training graph transformations.
    - Build the gradient graph.
    - Reorder the outputs of the gradient graph to match the rule:
       - [user outputs, gradients in the order of model parameter inputs]

    Args:
        model: The forward only model.
        requires_grad: The set of model parameter names that require gradient.
        frozen_params: The set of model parameter names that are frozen.
        output_names: The list of user output names.

    Returns:
        A tuple of (training model, eval model).
    """

    if isinstance(output_names, str):
        output_names = [output_names]

    _move_initializers_to_inputs(model, requires_grad.union(frozen_params))

    # At this point, eval model and training model diverge.
    eval_model = copy.deepcopy(model)
    _disable_training_mode(eval_model)

    options = SessionOptions()
    if custom_op_library is not None:
        options.register_custom_ops_library(os.fspath(custom_op_library))

    optimized_model = onnx.load_from_string(get_optimized_model(model.SerializeToString(), requires_grad, options))

    # Assumption is that the first graph output is the loss output
    gradient_model = _gradient_model_for(optimized_model, requires_grad, output_names[0], options)

    _reorder_outputs(gradient_model, output_names, requires_grad)

    return gradient_model, eval_model


def build_gradient_accumulation_graph(grad_model: onnx.ModelProto, requires_grad: Set[str]) -> None:
    """Builds gradient accumulation nodes on top of a training model.

    Adds an InPlaceAccumulatorV2 node for every gradient so that the gradients
    are accumulated in a gradient buffer (which is an input to InPlaceAccumulatorV2).

    For example, if there is a gradient in the graph called fc1.weight_grad,
    an InPlaceAccumulatorV2 will be added for that gradient whose input will
    be a graph input (fc1.weight_grad.accumulation.buffer) and the newly
    computed gradient (fc1.weight_grad).

    gradient.accumulation.buffer        gradient
        |         |                         |
        Ʌ         v                         v
        |         |_________________________|
        |                      |
        |               InPlaceAccumulatorV2
        |                      |
        |                      v
        |______________________|
    """

    # TODO: Avoid hard coded input/output strings
    gradient_output_names = {f"{arg_name}_grad" for arg_name in requires_grad}

    graph_inputs = grad_model.graph.input
    graph_nodes = grad_model.graph.node
    graph_outputs = []

    lazy_reset_grad_input_name = "lazy_reset_grad"
    gradient_accumulation_name = "accumulation"
    gradient_buffer_name_suffix = "buffer"
    gradient_output_name_suffix = "out"

    for idx, graph_output in enumerate(grad_model.graph.output):
        if graph_output.name not in gradient_output_names:
            # If graph output is not a gradient, there is no
            # need to build the gradient accumulator node for it.
            graph_outputs.append(graph_output)
            continue

        grad_name = graph_output.name
        grad_accumulation_buffer_name = f"{grad_name}.{gradient_accumulation_name}.{gradient_buffer_name_suffix}"
        grad_accumulation_output_name = f"{grad_name}.{gradient_accumulation_name}.{gradient_output_name_suffix}"

        # Gradient accumulation node
        acc_node = onnx.helper.make_node(
            "InPlaceAccumulatorV2",
            [grad_accumulation_buffer_name, grad_name, lazy_reset_grad_input_name],
            [grad_accumulation_output_name],
            name=f"GradientAccumulator{idx}",
            domain="com.microsoft",
        )

        graph_nodes.append(acc_node)

        # Grad buffer is also a graph input
        grad_accumulation_buffer_input = copy.deepcopy(graph_output)
        grad_accumulation_buffer_input.name = grad_accumulation_buffer_name
        graph_inputs.append(grad_accumulation_buffer_input)

        # Accumulated gradient update flag is also a graph output
        grad_accumulation_output = onnx.helper.make_tensor_value_info(
            grad_accumulation_output_name, onnx.TensorProto.BOOL, [1]
        )
        graph_outputs.append(grad_accumulation_output)

    lazy_reset_grad_input = onnx.helper.make_tensor_value_info(lazy_reset_grad_input_name, onnx.TensorProto.BOOL, [1])
    graph_inputs.append(lazy_reset_grad_input)

    del grad_model.graph.output[:]
    grad_model.graph.output.extend(graph_outputs)


def get_model_parameters(
    model: onnx.ModelProto, requires_grad: Set[str], frozen_params: Set[str]
) -> Tuple[List[onnx.TensorProto], List[onnx.TensorProto]]:
    """Returns trainable and non trainable onnx model parameters.

    This function pulls out the model parameters from the initializers in the graph.
    Initializers can be bucketed into three categories:
    - Trainable parameters: These are the parameters that require gradient.
    - Non trainable parameters: These are the parameters that do not require gradient.
    - Constants: These are parameters local to the graph and are not part of the model parameters as defined
    by the exporting framework.

    As of now, there is no between the three categories. For this reason, we keep the non trainable parameters
    embedded within the graph and do not expose them as model parameters.

    Args:
        model: The onnx model.
        requires_grad: The set of model parameter names that require gradient.
        frozen_params: The set of model parameter names that are frozen.

    Returns:
        A tuple of (trainable parameters, non trainable parameters).
    """

    trainable_params = []
    non_trainable_params = []
    for initializer in model.graph.initializer:
        if initializer.name in requires_grad:
            trainable_params.append(initializer)
        elif initializer.name in frozen_params:
            non_trainable_params.append(initializer)

    return trainable_params, non_trainable_params
