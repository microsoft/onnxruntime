
from abc import ABC, abstractmethod
from onnxruntime.capi._pybind_state import GradientGraphBuilder
import onnx
import copy
from .onnxflow_pb2 import OnnxFlowParameter, OnnxFlowParameters

def _build_gradient_model(model, requires_grad_params, frozen_params):
    # Collect names of parameters that need gradients computed
    trainable_parameters = set()
    # Move all trainable and non trainable initializers to graph inputs.
    # This allows training to pass in the parameters from outside the graph
    # so as to share the parameters across multiple sessions.
    graph_inputs = model.graph.input
    initializers = []
    for initializer in model.graph.initializer:
        if not initializer.name[0].isdigit():
            # Move onl those initializers as inputs that are not local
            # to the onnx model. i.e. initializers that are model parameters.
            # These are tpically those initializers without any number prefixed
            # to their names.
            graph_inputs.append(
                onnx.helper.make_tensor_value_info(initializer.name,
                                                   initializer.data_type,
                                                   initializer.dims))
            if initializer.name not in frozen_params:
                trainable_parameters.add(initializer.name)
        else:
            # All other initializers stay where they were.
            initializers.append(initializer)

    # Graph and model with initializers as inputs.
    graph_with_initializers_as_inputs = onnx.helper.make_graph(model.graph.node,
                                                               'graph_with_initializers_as_inputs',
                                                               graph_inputs, model.graph.output,
                                                               initializer=initializers)
    grad_model = onnx.helper.make_model(graph_with_initializers_as_inputs,
                                        producer_name='onnxflow',
                                        opset_imports=[
                                            onnx.helper.make_opsetid('com.microsoft', 1)] + \
                                            list(model.opset_import))

    # Any parameter or input that requires gradient, should have been already added to
    # requires_grad_params
    for parameter_name in requires_grad_params:
        trainable_parameters.add(parameter_name)

    # Assumption is that the graph has an output called `loss`.
    builder = GradientGraphBuilder(grad_model.SerializeToString(),
                                   {'loss'},
                                   trainable_parameters,
                                   'loss')
    builder.build()
    return onnx.load_from_string(builder.get_model())

def _build_gradient_accumulation_model(grad_model):
    graph_inputs = grad_model.graph.input
    graph_nodes = grad_model.graph.node
    graph_outputs = grad_model.graph.output
    for idx, graph_output in enumerate(grad_model.graph.output):
        # if the graph output ends with _grad,
        # assume that that output is a gradient output
        if not graph_output.name.endswith('_grad'):
            continue

        # gradient accumulation node inputs and output names
        grad_name = graph_output.name
        grad_accumulation_buffer_name = f'{grad_name}.accumulation.buffer'
        grad_accumulation_output_name = f'{grad_name}.accumulation.out'

        # Gradient accumulation node
        acc_node = onnx.helper.make_node("InPlaceAccumulator",
                                [grad_accumulation_buffer_name, grad_name],
                                [grad_accumulation_output_name],
                                name=f"GradientAccumulator{idx}",
                                domain='com.microsoft')

        graph_nodes.append(acc_node)

        # grad buffer is also a graph input
        grad_accumulation_buffer_input = copy.deepcopy(graph_output)
        grad_accumulation_buffer_input.name = grad_accumulation_buffer_name
        graph_inputs.append(grad_accumulation_buffer_input)

        # accumulated gradient is also a graph output
        grad_accumulation_output = copy.deepcopy(graph_output)
        grad_accumulation_output.name = grad_accumulation_output_name
        graph_outputs.append(grad_accumulation_output)

    graph = onnx.helper.make_graph(graph_nodes, 'GradientGraph',
                                   graph_inputs,
                                   graph_outputs,
                                   grad_model.graph.initializer)
    return onnx.helper.make_model(graph, producer_name='onnxflow',
                                  opset_imports=list(grad_model.opset_import))


def _get_model_parameters(model, requires_grad_params, frozen_params):
    parameters = OnnxFlowParameters()
    for initializer in model.graph.initializer:
        if not initializer.name[0].isdigit():
            param = OnnxFlowParameter()
            param.requires_grad = True
            if initializer.name in frozen_params:
                param.requires_grad = False
            param.data.Pack(initializer)
            param.is_parameter = True
            parameters.parameters.append(param)

    requires_grad_params_set = set(requires_grad_params)
    for graph_input in model.graph.input:
        if graph_input.name in requires_grad_params_set:
            param = OnnxFlowParameter()
            param.requires_grad = True
            param.data.Pack(graph_input)
            param.is_parameter = False
            parameters.parameters.append(param)

    return parameters


class Graph(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        # build the user model
        user_model = self.build(*args, **kwargs)

        # validate and check the model
        onnx.checker.check_model(user_model, True)

        return user_model

class TrainingGraph(Graph):
    def __init__(self):
        super(TrainingGraph, self).__init__()
        self._frozen = set()
        self._requires_grad = []
        self._parameters = None

    @abstractmethod
    def build(self, *args, **kwargs):
        ...

    def freeze_parameter(self, parameter_name):
        self._frozen.add(parameter_name)

    def requires_grad(self, parameter_name):
        self._requires_grad.append(parameter_name)

    def parameters(self):
        # return parameters that can be serialized by the user
        if self._parameters is None:
            raise RuntimeError("Please build the training graph first before trying to retrieve the parameters.")

        return self._parameters

    def __call__(self, *args, **kwargs):
        # build the user model
        user_model = self.build(*args, **kwargs)

        # get all the model parameters for the user_model
        self._parameters = _get_model_parameters(user_model, self._requires_grad, self._frozen)

        # build the gradient graph
        grad_model = _build_gradient_model(user_model, self._requires_grad, self._frozen)

        # add gradient accumulation nodes
        grad_model = _build_gradient_accumulation_model(grad_model)

        # validate and check the model
        onnx.checker.check_model(grad_model, True)

        return grad_model
