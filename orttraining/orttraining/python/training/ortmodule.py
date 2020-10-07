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
        self._onnx_forward_initializers_desc = []
        self._onnx_forward_inputs_desc = []
        self._onnx_forward_outputs_desc = []

        # Backward pass
        self._onnx_backward = None

    def forward(self, *input, **kwargs):
        print(f'ORTModule.forward() was called')

        if not self._onnx_forward:
            original_forward_graph = ORTModule._get_forward_graph(self._original_module, *input, **kwargs)
            # gradient_graph = ORTModule._build_gradient_graph(original_forward_graph)
            # self.forward_graph, self.backward_graph = ORTModule._split_forward_and_backward(gradient_graph)
            self._onnx_forward = original_forward_graph # TODO: hard-coding for MVP
            # import pdb; pdb.set_trace()
            self.forward_session = onnxruntime.InferenceSession(self._onnx_forward.SerializeToString())


            # TrainingParameters
            # ort_parameters = onnxruntime.TrainingParameters()
            # ort_parameters.loss_output_name = "loss"
            # ort_parameters.use_mixed_precision = False
            # ort_parameters.world_rank = 0
            # ort_parameters.world_size = 1
            # ort_parameters.gradient_accumulation_steps = 1
            # ort_parameters.allreduce_post_accumulation = False
            # ort_parameters.deepspeed_zero_stage = 0
            # ort_parameters.enable_grad_norm_clip = False
            # ort_parameters.set_gradients_as_graph_outputs = False
            # ort_parameters.use_invertible_layernorm_grad = False
            # ort_parameters.training_optimizer_name = "SGDOptimizer"
            # ort_parameters.lr_params_feed_name = "Learning_Rate"
            # ort_parameters.weights_to_train = trainable_params
            # ort_parameters.optimizer_attributes_map = optimizer_attributes_map
            # ort_parameters.optimizer_int_attributes_map = optimizer_int_attributes_map

            # # SessionOptions
            # session_options = onnxruntime.SessionOptions()
            # session_options.use_deterministic_compute = self.options.debug.deterministic_compute
            # self.forward_session = onnxruntime.TrainingSession(self._onnx_forward.SerializeToString(), ort_parameters, session_options)

            self._save_onnx_graph(self._onnx_forward, 'forward_mnist.onnx')
        if not self._onnx_forward_initializers_desc:
            self._onnx_forward_initializers_desc = self._get_initializer_from_graph(self._onnx_forward)
        if not self._onnx_forward_inputs_desc:
            self._onnx_forward_inputs_desc = self._get_input_from_graph(self._onnx_forward)
        if not self._onnx_forward_outputs_desc:
            self._onnx_forward_outputs_desc = self._get_output_from_graph(self._onnx_forward)

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
                # import pdb; pdb.set_trace()
                outputs = self._run_forward_graph(*input, **kwargs) # inputs, weights)
                outputs = [torch.from_numpy(out).requires_grad_(True) for out in outputs]

                # TODO: Properly save intermediate tensors and remove them from model output
                ctx.save_for_backward(outputs[1])
                outputs = [outputs[0]]

                # TODO: Properly support original module output format
                if len(outputs) == 1:
                    return outputs[0]
                return tuple(outputs)

            @staticmethod
            def backward(ctx, grad_output):
                print(f'_ORTModuleFunction.backward() was called')
                ...
                # intermediates = ctx.saved_tensors
                # grad_inputs, grad_weights = self._run_backward_graph(
                #     grad_output, intermediates)
                # return grad_inputs, grad_weights

        return _ORTModuleFunction.apply(self._prepare_model_input(*input, **kwargs))

    def _prepare_model_input(self, *input, **kwargs):
        # Dictionary containing both inputs and initializers
        input_with_initializer = {}

        # import pdb; pdb.set_trace()
        # Inputs
        for idx, input_data in enumerate(self.forward_session.get_inputs()):
            input_with_initializer.update({input_data.name : input[idx].cpu().numpy()})

        # Initializers
        for idx, param in enumerate(self._original_module.named_parameters()):
            input_with_initializer.update({param[0] : param[1].detach().numpy()})

        return input_with_initializer

    def _run_forward_graph(self, data_with_initializer): #input, weights):
        # import pdb; pdb.set_trace()
        return self.forward_session.run(None, data_with_initializer)

    def _run_backward_graph(self, grad_output, intermediates):
        # Use an InferenceSession to execute self.backward_graph.
        # Return gradient tensors for inputs and weights.
        ...

    @staticmethod
    def _get_forward_graph(module, module_input):
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

    def _get_initializer_from_graph(self, graph):
        # TODO: There is a tradefoo between memory footprint and total model export time
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
            initializers.append({'name' : name, 'shape' : shape, 'dtype' : dtype})
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
                inputs.append({'name' : name, 'shape' : shape, 'dtype' : dtype})
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
                outputs.append({'name' : name, 'shape' : shape, 'dtype' : dtype})
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
        # Invoke the C++ GradientBuilder implementation via pybind.
        # Return an ONNX graph that contains the forward and backward nodes, which takes the
        # following inputs:
        # * Module inputs
        # * Module weights
        # * Gradients with respect to the module outputs
        # …and produces gradients with respect to the module inputs and weights.
        ...

    @staticmethod
    def _split_forward_and_backward(gradient_graph):
        # Split the result of _build_gradient_graph into two subgraphs:
        # * A forward graph that takes module inputs and weights as input, and produces module
        #   outputs and (“stashed”) intermediate tensors as output.
        # * A backward graph that takes intermediate tensors, module weights, and gradients
        #   respect to the module outputs as inputs, and produces gradients with respect to the
        #   module inputs and weights.
        return (None, None)
