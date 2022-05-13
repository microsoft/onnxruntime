import copy
from onnxruntime.capi._pybind_state import GradientGraphBuilder
import onnx

from onnxruntime.training import onnxblock


def build_gradient_model(model, user_args_requiring_grad, user_args_not_requiring_grad):
    """Builds the gradient graph on top of the given input forward only graph."""

    # Collect names of parameters that need gradients computed
    all_args_requiring_gradient = set()
    # Move all trainable and non trainable initializers to graph inputs.
    # This allows training to pass in the parameters from outside the graph
    # so as to share the parameters across multiple sessions.
    graph_inputs = model.graph.input
    initializers = []
    for initializer in model.graph.initializer:
        if not initializer.name[0].isdigit():
            # Move only those initializers as inputs that are not local
            # to the onnx model. i.e. initializers that are model parameters.
            # These are tpically those initializers without any number prefixed
            # to their names.
            graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    initializer.name, initializer.data_type, initializer.dims
                )
            )
            if initializer.name not in user_args_not_requiring_grad:
                all_args_requiring_gradient.add(initializer.name)
        else:
            # All other initializers stay where they were.
            initializers.append(initializer)

    # Graph and model with initializers as inputs.
    graph_with_initializers_as_inputs = onnx.helper.make_graph(
        model.graph.node,
        "Forward Graph with Initilializers as Inputs",
        graph_inputs,
        model.graph.output,
        initializer=initializers,
    )
    grad_model = onnx.helper.make_model(
        graph_with_initializers_as_inputs,
        producer_name=onnxblock._producer_name,
        opset_imports=[onnx.helper.make_opsetid("com.microsoft", 1)]
        + list(model.opset_import),
    )

    # Any graph input that requires gradient, should have been already added to
    # args_requiring_grad.
    for argument_name in user_args_requiring_grad:
        all_args_requiring_gradient.add(argument_name)

    # Assumption is that the graph has an output called `loss`.
    builder = GradientGraphBuilder(
        grad_model.SerializeToString(), {"loss"}, all_args_requiring_gradient, "loss"
    )
    builder.build()
    return onnx.load_from_string(builder.get_model())


def build_gradient_accumulation_model(grad_model):
    """Builds gradient accumulation nodes on top of a training model."""

    graph_inputs = grad_model.graph.input
    graph_nodes = grad_model.graph.node
    graph_outputs = []

    lazy_reset_grad_input_name = "lazy_reset_grad"
    gradient_output_suffix = "_grad"
    gradient_accumulation_name = "accumulation"
    gradient_buffer_name_suffix = "buffer"
    gradient_output_name_suffix = "out"

    for idx, graph_output in enumerate(grad_model.graph.output):
        # if the graph output ends with _grad,
        # assume that that output is a gradient output
        if not graph_output.name.endswith(gradient_output_suffix):
            graph_outputs.append(graph_output)
            continue

        # gradient accumulation node inputs and output names
        grad_name = graph_output.name
        grad_accumulation_buffer_name = (
            f"{grad_name}.{gradient_accumulation_name}.{gradient_buffer_name_suffix}"
        )
        grad_accumulation_output_name = (
            f"{grad_name}.{gradient_accumulation_name}.{gradient_output_name_suffix}"
        )

        # Gradient accumulation node
        acc_node = onnx.helper.make_node(
            "InPlaceAccumulator",
            [grad_accumulation_buffer_name, grad_name, lazy_reset_grad_input_name],
            [grad_accumulation_output_name],
            name=f"GradientAccumulator{idx}",
            domain="com.microsoft",
        )

        graph_nodes.append(acc_node)

        # grad buffer is also a graph input
        grad_accumulation_buffer_input = copy.deepcopy(graph_output)
        grad_accumulation_buffer_input.name = grad_accumulation_buffer_name
        graph_inputs.append(grad_accumulation_buffer_input)

        # accumulated gradient is also a graph output
        grad_accumulation_output = copy.deepcopy(graph_output)
        grad_accumulation_output.name = grad_accumulation_output_name
        graph_outputs.append(grad_accumulation_output)

    lazy_reset_grad_input = onnx.helper.make_tensor_value_info(
        lazy_reset_grad_input_name, onnx.TensorProto.BOOL, [1]
    )
    graph_inputs.append(lazy_reset_grad_input)

    graph = onnx.helper.make_graph(
        graph_nodes,
        "Training Graph with Gradient Accumulation Nodes",
        graph_inputs,
        graph_outputs,
        grad_model.graph.initializer,
    )
    return onnx.helper.make_model(
        graph,
        producer_name=onnxblock._producer_name,
        opset_imports=list(grad_model.opset_import),
    )


def get_model_parameters(model, args_not_requiring_gradient):
    """Returns trainable and non trainable onnx model parameters."""

    trainable_params = []
    non_trainable_params = []
    for initializer in model.graph.initializer:
        if not initializer.name[0].isdigit():
            if initializer.name in args_not_requiring_gradient:
                non_trainable_params.append(initializer)
            else:
                trainable_params.append(initializer)

    return trainable_params, non_trainable_params
