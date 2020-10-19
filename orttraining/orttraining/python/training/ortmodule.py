import copy
import io
import onnx
import onnxruntime
import os
import torch
import warnings

from . import _utils


ONNX_OPSET_VERSION = 12


class ORTModule(torch.nn.Module):

    def __init__(self, module):
        assert isinstance(module, torch.nn.Module), "'module' mst be a torch.nn.Module"
        super(ORTModule, self).__init__()

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module

        # Forward pass
        self._onnx_forward = None
        self._forward_session = None
        self._onnx_forward_initializers_desc = []
        self._onnx_forward_inputs_desc = []
        self._onnx_forward_outputs_desc = []

        # Backward pass
        self._onnx_backward = None
        self._backward_session = None
        self._onnx_backward_initializers_desc = []
        self._onnx_backward_inputs_desc = []
        self._onnx_backward_outputs_desc = []

    def forward(self, *input, **kwargs):
        if not self._onnx_forward:
            original_forward_graph = ORTModule._get_forward_graph(self._original_module, *input, **kwargs)
            gradient_graph = ORTModule._build_gradient_graph(original_forward_graph)
            # TODO: Remove manual split after MVP
            # self.forward_graph, self.backward_graph = ORTModule._split_forward_and_backward(gradient_graph)
            self._onnx_forward = original_forward_graph  # TODO: hard-coding for MVP
            self._onnx_backward = gradient_graph  # TODO: hard-coding for MVP
            self._forward_session = onnxruntime.InferenceSession(self._onnx_forward.SerializeToString())
            self._backward_session = onnxruntime.InferenceSession(self._onnx_backward.SerializeToString())

        # Forward I/O description
        if not self._onnx_forward_initializers_desc:
            self._onnx_forward_initializers_desc = self._get_initializer_from_graph(self._onnx_forward)
            print(f'Forward initializers: {self._onnx_forward_initializers_desc}')
        if not self._onnx_forward_inputs_desc:
            self._onnx_forward_inputs_desc = self._get_input_from_graph(self._onnx_forward)
            print(f'Forward inputs: {self._onnx_forward_inputs_desc}')
        if not self._onnx_forward_outputs_desc:
            self._onnx_forward_outputs_desc = self._get_output_from_graph(self._onnx_forward)
            print(f'Forward outputs: {self._onnx_forward_outputs_desc}')

        # Backward I/O description
        if not self._onnx_backward_initializers_desc:
            self._onnx_backward_initializers_desc = self._get_initializer_from_graph(self._onnx_backward)
            print(f'Backward initializers: {self._onnx_backward_initializers_desc}')
        if not self._onnx_backward_inputs_desc:
            self._onnx_backward_inputs_desc = self._get_input_from_graph(self._onnx_backward)
            print(f'Backward inputs: {self._onnx_forward_inputs_desc}')
        if not self._onnx_backward_outputs_desc:
            self._onnx_backward_outputs_desc = self._get_output_from_graph(self._onnx_backward)
            print(f'Backward outputs: {self._onnx_backward_outputs_desc}')

        # Use a custom torch.autograd.Function to associate self.backward_graph as the
        # gradient implementation for self.forward_graph.
        class _ORTModuleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *input, **kwargs):
                # TODO: Potential optimization is to detect which inputs and weights require gradients
                input_with_initializer = self._prepare_forward_input_ort(*input, **kwargs)
                outputs = self._run_forward_graph(input_with_initializer)
                outputs = tuple(torch.from_numpy(out) for out in outputs)

                # TODO: Properly save dynamic number of intermediate tensors and remove them from model output
                #       Tensors that need to have gradients tracked can't be saved by `save_for_backward`
                #       saved_tensors ==> input1, fc2.weight, 7
                ctx.save_for_backward(*[input[0], input[3], outputs[1]])
                outputs = [outputs[0]]

                # TODO: Properly support original module output format
                if len(outputs) == 1:
                    return outputs[0]
                return outputs

            @staticmethod
            def backward(ctx, *grad_output):
                # TODO: Properly restore dynamic number of intermediate tensors
                #       saved_tensors ==> input1, fc2.weight, 7
                saved_tensors = ctx.saved_tensors
                grad_weights = self._run_backward_graph(*[*saved_tensors, *grad_output])
                grad_weights = [torch.from_numpy(grad) for grad in grad_weights]
                # TODO: backward must return grad tensors in the same order forward does
                #       [input1_grad, fc1.weight_grad, fc1.bias_grad, fc2.weight_grad, fc2.bias_grad]
                return tuple([torch.tensor([1.]), grad_weights[1], grad_weights[0], grad_weights[2], grad_weights[3]])

        return _ORTModuleFunction.apply(*self._prepare_forward_input_autograd(*input, **kwargs))

    def _prepare_forward_input_autograd(self, *input, **kwargs):
        # List containing both user inputs and initializers, in this order
        input_with_initializer = []

        # Inputs
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            input_with_initializer.append(input[idx])

        # Initializers
        for idx, param in enumerate(self._original_module.named_parameters()):
            input_with_initializer.append(param[1])

        # TODO: [input1, fc1.weight, fc1.bias, fc2.weight, fc2.bias]
        return input_with_initializer

    def _prepare_forward_input_ort(self, *inputs):
        # Dictionary containing both inputs and initializers
        input_with_initializer = {}

        # TODO: [input1, fc1.weight, fc1.bias, fc2.weight, fc2.bias]
        # Inputs
        inputs_len = 0
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            inputs_len += 1
            input_with_initializer.update({input_data.name: inputs[idx].cpu().numpy()})

        # Initializers
        for param in self._original_module.named_parameters():
            input_with_initializer.update({param[0]: inputs[inputs_len].detach().numpy()})
            inputs_len += 1

        return input_with_initializer

    def _prepare_backward_input(self, *inputs, **kwargs):
        # Dictionary containing initializers
        input_with_initializer = {}

        # User input
        # TODO: How to determine which user input to feed to backward
        # for idx, input_data in enumerate(self._forward_session.get_inputs()):
        #     input_with_initializer.update({input_data.name: inputs[idx].cpu().numpy()})
        input_with_initializer.update({'input1' : inputs[0].detach().numpy()})

        # Initializers
        # TODO: How to determine which initializer (subset) to be used
        # for idx, param in enumerate(self._original_module.named_parameters()):
        #     input_with_initializer.update({param[0]: param[1].detach().numpy()})
        input_with_initializer.update({'fc2.weight' : inputs[1].detach().numpy()})

        # Intermediates
        # TODO: How to determine intermediates name?
        input_with_initializer.update({'7': inputs[2].detach().numpy()})

        # Grad output
        # TODO: How to determine grad_output name?
        input_with_initializer.update({'probability_grad': inputs[3].detach().numpy()})
        return input_with_initializer

    def _run_forward_graph(self, data_with_initializer):  # input, weights):
        return self._forward_session.run(None, data_with_initializer)

    def _run_backward_graph(self, *inputs, **kwargs):
        data = self._prepare_backward_input(*inputs, **kwargs)
        # TODO: Hack to guarantee output order from InferenceSession.run()
        return self._backward_session.run(['fc1.bias_grad', 'fc1.weight_grad', 'fc2.weight_grad', 'fc2.bias_grad'], data)

    @staticmethod
    def _get_forward_graph(module, module_input):
        # TODO: Pytorch module must be exported to ONNX and splitted
        #       Hard-coding with MNIST stub for MVP
        return onnx.load('./model_with_training_forward_sliced.onnx')

    def _get_initializer_from_graph(self, graph):
        # TODO: There is a tradeoff between memory footprint and total model export time
        #   Ideally we want to export the model using torch.onnx.export(.., export_params=False, keep_initializers_as_inputs=True)
        #   to obtain an ONNX model with minimal size and initializers as input.
        #   However, this results in (guessing) assuming only initializer's name end with '.weight' and '.bias'.
        #   Otherwise, it is not possible to separate input from initializer after the model is exported
        #   Options are:
        #   1) If memory footprint is more important, we can export ONNX twice, varying keep_initializers_as_inputs flag
        #       ONNX model is small (400 bytes vs 1.6MB for MNIST), but export takes twice the time
        #   2) If total export time is more important, we can export ONNX once, using export_params=True
        #       ONNX model is bigger, but export takes half the time

        # As performance is not the main goal in this first deliverable, using approach 2) for simplicity
        initializers = []
        for initializer in graph.graph.initializer:
            name = initializer.name
            # TODO: Dynamic shape is not being handled yet
            shape = initializer.dims
            dtype = _utils.dtype_onnx_to_torch(initializer.data_type)
            initializers.append({'name': name, 'shape': shape, 'dtype': dtype})
        return initializers

    def _get_input_from_graph(self, graph):
        inputs = []
        for elem in graph.graph.input:
            for initializer in self._onnx_forward_initializers_desc:
                if elem.name == initializer['name']:
                    break
            else:
                name = elem.name
                # TODO: Dynamic shape is not being handled yet
                shape = [dim.dim_value for dim in elem.type.tensor_type.shape.dim]
                dtype = _utils.dtype_onnx_to_torch(elem.type.tensor_type.elem_type)
                inputs.append({'name': name, 'shape': shape, 'dtype': dtype})
        return inputs

    def _get_output_from_graph(self, graph):
        outputs = []
        for elem in graph.graph.output:
            for initializer in self._onnx_forward_initializers_desc:
                if elem.name == initializer['name']:
                    # skip initializers
                    break
            else:
                name = elem.name
                # TODO: Dynamic shape is not being handled yet
                shape = [dim.dim_value for dim in elem.type.tensor_type.shape.dim]
                dtype = _utils.dtype_onnx_to_torch(elem.type.tensor_type.elem_type)
                outputs.append({'name': name, 'shape': shape, 'dtype': dtype})
        return outputs

    @staticmethod
    def _save_onnx_graph(onnx_graph, path):
        r"""Persists ONNX model into :py:attr:`path`

        The model will be saved as a Google Protocol Buffers (aka protobuf) file as per ONNX standard.
        The graph includes full information, including inference and training metadata.

        Args:
            onnx_graph (onnx.ModelProto): Either forward or backward graph
            path (str): Full path, including filename, to save the ONNX model in the filesystem

        Raises:
            ValueError: raised when `path` is not valid path
        """

        assert isinstance(path, str), "'path' must be a valid path string"
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path)
        if (dir_name and not os.path.exists(dir_name)) or not file_name:
            warnings.warn("'path' is not valid or does not exist")
            return

        with open(path, "wb") as f:
            f.write(onnx_graph.SerializeToString())

    @staticmethod
    def _build_gradient_graph(forward_graph):
        # TODO: Invoke the C++ GradientBuilder implementation via pybind.
        # Return an ONNX graph that contains the forward and backward nodes, which takes the
        # following inputs:
        # * Module inputs
        # * Module weights
        # * Gradients with respect to the module outputs
        # …and produces gradients with respect to the module inputs and weights.
        return onnx.load('./model_with_training_backward_sliced.onnx')

    @staticmethod
    def _split_forward_and_backward(gradient_graph):
        # TODO: Split the result of _build_gradient_graph into two subgraphs:
        # * A forward graph that takes module inputs and weights as input, and produces module
        #   outputs and (“stashed”) intermediate tensors as output.
        # * A backward graph that takes intermediate tensors, module weights, and gradients
        #   respect to the module outputs as inputs, and produces gradients with respect to the
        #   module inputs and weights.
        return (None, None)
