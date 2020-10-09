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
        print(f'ORTModule.__init__() was called')
        super(ORTModule, self).__init__()
        # User will interact with it (debugging, etc)
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

    def forward(self, *input, **kwargs):
        print(f'ORTModule.forward() was called')

        if not self._onnx_forward:
            original_forward_graph = ORTModule._get_forward_graph(self._original_module, *input, **kwargs)
            gradient_graph = ORTModule._build_gradient_graph(original_forward_graph)
            # TODO: Remove manual split after MVP
            # self.forward_graph, self.backward_graph = ORTModule._split_forward_and_backward(gradient_graph)
            self._onnx_forward = original_forward_graph  # TODO: hard-coding for MVP
            self._onnx_backward = gradient_graph  # TODO: hard-coding for MVP
            self._forward_session = onnxruntime.InferenceSession(self._onnx_forward.SerializeToString())
            self._backward_session = onnxruntime.InferenceSession(self._onnx_backward.SerializeToString())

            # TODO: debug only
            self._save_onnx_graph(self._onnx_forward, 'ortmodule_forward_mnist.onnx')
            self._save_onnx_graph(self._onnx_backward, 'ortmodule_backward_mnist.onnx')

        if not self._onnx_forward_initializers_desc:
            self._onnx_forward_initializers_desc = self._get_initializer_from_graph(self._onnx_forward)
        if not self._onnx_forward_inputs_desc:
            self._onnx_forward_inputs_desc = self._get_input_from_graph(self._onnx_forward)
        if not self._onnx_forward_outputs_desc:
            self._onnx_forward_outputs_desc = self._get_output_from_graph(self._onnx_forward)

        # TODO: debug only
        print(f'Initializers: {self._onnx_forward_initializers_desc}')
        print(f'Inputs: {self._onnx_forward_inputs_desc}')
        print(f'Outpus: {self._onnx_forward_outputs_desc}')

        # Use a custom torch.autograd.Function to associate self.backward_graph as the
        # gradient implementation for self.forward_graph.
        class _ORTModuleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *input, **kwargs):
                print(f'_ORTModuleFunction.forward() was called...')
                # Note: A potential optimization would be to detect which of inputs and weights
                # require a gradient.
                # intermediates, outputs = self._run_forward_graph(inputs) # inputs, weights)
                outputs = self._run_forward_graph(*input, **kwargs)  # inputs, weights)
                outputs = [torch.from_numpy(out).requires_grad_(True) for out in outputs]

                # TODO: Properly save intermediate tensors and remove them from model output
                ctx.save_for_backward([(input, kwargs), outputs[1]])
                # outputs = [outputs[0]]

                # TODO: Properly support original module output format
                if len(outputs) == 1:
                    return outputs[0]
                return tuple(outputs)

            @staticmethod
            def backward(ctx, *grad_output):
                print(f'_ORTModuleFunction.backward() was called')
                input_and_kwargs, intermediates = ctx.saved_tensors
                # grad_inputs, grad_weights = self._run_backward_graph(
                #     grad_output, intermediates)
                # return grad_inputs, grad_weights

        return _ORTModuleFunction.apply(self._prepare_forward_input(*input, **kwargs))

    def _prepare_forward_input(self, *input, **kwargs):
        # Dictionary containing both inputs and initializers
        input_with_initializer = {}

        # Inputs
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            input_with_initializer.update({input_data.name: input[idx].cpu().numpy()})

        # Initializers
        for idx, param in enumerate(self._original_module.named_parameters()):
            input_with_initializer.update({param[0]: param[1].detach().numpy()})

        return input_with_initializer

    def _prepare_backward_input(self, grad_output, intermediates, *inputs, **kwargs):
        # Dictionary containing initializers
        input_with_initializer = {}

        # User input
        # TODO: How to determine which user input to feed to backward
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            input_with_initializer.update({input_data.name: inputs[idx].cpu().numpy()})

        # Initializers
        # TODO: How to determine which initializer (subset) to be used
        for idx, param in enumerate(self._original_module.named_parameters()):
            if param[0] == 'fc2.weight':
                input_with_initializer.update({param[0]: param[1].detach().numpy()})

        # Grad output
        # TODO: How to determine grad_output name?
        input_with_initializer.update({'probability_grad': grad_output.detach().numpy()})

        # Intermediates
        # TODO: How to determine intermediates name?
        input_with_initializer.update({'7': intermediates.detach().numpy()})
        return input_with_initializer

    def _run_forward_graph(self, data_with_initializer):  # input, weights):
        print(f'_run_forward_graph was called...')
        return self._forward_session.run(None, data_with_initializer)

    def _run_backward_graph(self, grad_output, intermediates, *inputs, **kwargs):
        # Use an InferenceSession to execute self.backward_graph.
        # Return gradient tensors for inputs and weights.
        print(f'_run_backward_graph was called...')
        data = self._prepare_backward_input(grad_output, intermediates, *inputs, **kwargs)
        return self._backward_session.run(None, data)

    @staticmethod
    def _get_forward_graph(module, module_input):
        print(f'_get_forward_graph was called...')
        # TODO: Pytorch module must be exported to ONNX and splitted
        #       Hard-coding with MNIST stub for MVP
        # Export torch.nn.Module to ONNX with initializers as input
        # f = io.BytesIO()
        # torch.onnx.export(module, module_input, f, verbose=True,
        #                   opset_version=ONNX_OPSET_VERSION,
        #                   _retain_param_name=True,
        #                   training=torch.onnx.TrainingMode.TRAINING,
        #                   keep_initializers_as_inputs=True,
        #                   export_params=True)
        # return onnx.load_model_from_string(f.getvalue())
        return onnx.load('/home/thiagofc/mnist_onnx/mnist_with_training_forward_sliced.onnx')
        # return onnx.load('/home/thiagofc/mnist_onnx/mnist_with_training.onnx')

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
        print(f'_build_gradient_graph was called...')
        # TODO: Invoke the C++ GradientBuilder implementation via pybind.
        # Return an ONNX graph that contains the forward and backward nodes, which takes the
        # following inputs:
        # * Module inputs
        # * Module weights
        # * Gradients with respect to the module outputs
        # …and produces gradients with respect to the module inputs and weights.
        return onnx.load('/home/thiagofc/mnist_onnx/mnist_with_training_backward_sliced.onnx')

    @staticmethod
    def _split_forward_and_backward(gradient_graph):
        print(f'_split_forward_and_backward was called...')
        # TODO: Split the result of _build_gradient_graph into two subgraphs:
        # * A forward graph that takes module inputs and weights as input, and produces module
        #   outputs and (“stashed”) intermediate tensors as output.
        # * A backward graph that takes intermediate tensors, module weights, and gradients
        #   respect to the module outputs as inputs, and produces gradients with respect to the
        #   module inputs and weights.
        return (None, None)
