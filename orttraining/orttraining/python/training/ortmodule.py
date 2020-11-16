import copy
import io
import logging
import onnx
import onnxruntime
import os
import torch
import warnings
import numpy as np
from inspect import signature

from onnxruntime.capi import _pybind_state as C
from . import _utils


ONNX_OPSET_VERSION = 12


def get_device_index(device):
    if type(device) == str:
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        device = torch.device(device)
    return 0 if device.index is None else device.index


def prepare_io_binding(io_binding, inputs, model, output_buffers, device):
    idx = 0
    for value_info in model.graph.input:
        io_binding.bind_input(value_info.name, inputs[idx].device.type, get_device_index(inputs[idx].device),
                              _utils.dtype_torch_to_numpy(inputs[idx].dtype), list(inputs[idx].size()),
                              inputs[idx].data_ptr())
        idx += 1

    for value_info in model.graph.output:
        name = value_info.name
        output_tensor = output_buffers[name]
        io_binding.bind_output(name, output_tensor.device.type, get_device_index(device),
                               _utils.dtype_torch_to_numpy(output_tensor.dtype), list(output_tensor.size()),
                               output_tensor.data_ptr())


def value_info_to_buffer_tensor(value_info, device):
    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    dtype = _utils.dtype_onnx_to_torch(value_info.type.tensor_type.elem_type)
    return torch.zeros(shape, device=device, dtype=dtype)


class ORTModule(torch.nn.Module):

    def __init__(self, module, device="cpu", use_iobinding=False):
        assert isinstance(module, torch.nn.Module), "'module' mst be a torch.nn.Module"
        super(ORTModule, self).__init__()

        self._device = device
        self._use_iobinding = use_iobinding

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        self._onnx_training = None
        self._onnx_gradient = None

        # Forward pass
        self._onnx_forward = None
        self._forward_session = None

        # Backward pass
        self._onnx_backward = None
        self._backward_session = None

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # TODO: debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''


    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, a full training graph is splitted in forward and backward graph which are used
        to instantiate ONNX Runtime InferenceSession`s
        '''
        if not self._onnx_forward:
            self._onnx_training = ORTModule._get_forward_graph(self._original_module, *inputs, **kwargs)
            grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()
            # TODO: PyTorch exporter bug: changes the initializer order
            # Use the order in original module
            initializer_names = [p[0] for p in self._original_module.named_parameters()]
            self._onnx_gradient, self._onnx_forward, self._onnx_backward, self._onnx_graphs_info = ORTModule._build_fw_bw_grad_graphs(self._onnx_training, grad_builder_config, initializer_names)

            if self._save_onnx:
                onnx.save(self._onnx_training, self._save_onnx_prefix + '_full_training.onnx')
                onnx.save(self._onnx_gradient, self._save_onnx_prefix + '_with_grad.onnx')
                onnx.save(self._onnx_forward, self._save_onnx_prefix + '_forward.onnx')
                onnx.save(self._onnx_backward, self._save_onnx_prefix + '_backward.onnx')

            # TODO: Consider moving this to the backend. We don't want to append '_grad' to get correct tensor names
            self._onnx_graphs_types = ORTModule._get_io_info_from_onnx_graph(self._onnx_forward, self._onnx_graphs_info)

            execution_providers = ['CPUExecutionProvider'] if self._device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._forward_session = onnxruntime.InferenceSession(self._onnx_forward.SerializeToString(), providers=execution_providers)
            self._backward_session = onnxruntime.InferenceSession(self._onnx_backward.SerializeToString(), providers=execution_providers)

            if self._use_iobinding:
                self._forward_io_binding = self._forward_session.io_binding()
                self._forward_output_buffers = {}
                for output in self._onnx_forward.graph.output:
                    self._forward_output_buffers[output.name] = value_info_to_buffer_tensor(output, self._device)
                self._backward_io_binding = self._backward_session.io_binding()
                self._backward_output_buffers = {}
                for output in self._onnx_backward.graph.output:
                    self._backward_output_buffers[output.name] = value_info_to_buffer_tensor(output, self._device)

        # Use a custom torch.autograd.Function to associate self.backward_graph as the
        # gradient implementation for self.forward_graph.
        class _ORTModuleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                '''Performs forward pass based on user input and PyTorch initializer

                TODO: **kwargs are not supported

                Model outputs are returned to the user
                The following tensors are stashed (in order) for backward pass
                    * (Partial) user input
                    * (Partial) Initializers
                    * Intermediate tensors
                '''

                if not self._use_iobinding:
                    # Convert input to dict of torch tensors
                    data_dict = self._convert_forward_input_list_to_dict(*inputs)

                    # Convert dict of torch tensors to dict of numpy arrays (ORT BE requirement)
                    data_dict_numpy = self._convert_dict_torch_to_numpy(data_dict)

                    # Feed forward
                    outputs, intermediate = self._run_forward_graph(data_dict_numpy)
                    outputs = tuple(torch.from_numpy(item) for item in outputs)

                    # Save input, initializers and intermediate tensors to be used during backward
                    user_input = self._onnx_graphs_info.user_input_names
                    backward_user_input = self._onnx_graphs_info.backward_user_input_names
                    ctx_input = tuple(data_dict[name] for name in user_input if name in backward_user_input)
                    forward_initializer = self._onnx_graphs_info.initializer_names_to_train
                    backward_intializer = self._onnx_graphs_info.backward_intializer_names_as_input
                    ctx_initializer = tuple(data_dict[name] for name in forward_initializer if name in backward_intializer)
                    intermediate = tuple(torch.from_numpy(item) for item in intermediate)
                    ctx.save_for_backward(*[*ctx_input, *ctx_initializer, *intermediate])

                    # TODO: Support original module output (currently dict is not supported)
                    if len(outputs) == 1:
                        return outputs[0]
                    return outputs

                # Use IO binding.
                prepare_io_binding(self._forward_io_binding, inputs, self._onnx_forward, self._forward_output_buffers, self._device)
                self._forward_session.run_with_iobinding(self._forward_io_binding)

                forward_input_dict = self._convert_forward_input_list_to_dict(*inputs)
                ctx_inputs = tuple(forward_input_dict[name] for name in self._onnx_graphs_info.backward_user_input_names)
                ctx_initializers = tuple(forward_input_dict[name] for name in self._onnx_graphs_info.backward_intializer_names_as_input)
                ctx_intermediates = tuple(self._forward_output_buffers[name] for name in self._onnx_graphs_info.intermediate_tensor_names)
                ctx.save_for_backward(*[*ctx_inputs, *ctx_initializers, *ctx_intermediates])

                outputs = tuple(self._forward_output_buffers[name] for name in self._onnx_graphs_info.user_output_names)
                return outputs[0] if len(outputs) == 1 else outputs

            @staticmethod
            def backward(ctx, *grad_output):
                '''Performs backward pass based on grad wrt output and internal state

                Internal state is composed of:
                    * Tensor stashed (in a particular order) during forward:
                        * (partial) user input, (partial) initializers and intermediate tensors

                TODO: Input gradient is hard-coded to torch.tensor([1.])
                '''
                if not self._use_iobinding:
                    saved_tensors = ctx.saved_tensors
                    grad_weights = self._run_backward_graph(*[*saved_tensors, *grad_output])

                    result = [torch.tensor([1])]* len(self._onnx_graphs_info.user_input_names)
                    result += [torch.from_numpy(grad) for grad in grad_weights]
                    return tuple(result)

                # Use IO binding.
                grad_output_dict = dict(zip(self._onnx_graphs_info.user_output_grad_names, grad_output))
                backward_grad_output = tuple(grad_output_dict[name] for name in self._onnx_graphs_info.backward_output_grad_names)
                prepare_io_binding(self._backward_io_binding, [*ctx.saved_tensors, *backward_grad_output], self._onnx_backward, self._backward_output_buffers, self._device)
                self._backward_session.run_with_iobinding(self._backward_io_binding)

                results = [torch.tensor([1])] * len(self._onnx_graphs_info.user_input_names)
                results += [self._backward_output_buffers[name] for name in self._onnx_graphs_info.initializer_grad_names_to_train]
                return tuple(results)

        proc_inputs = [data for data in inputs if data is not None]
        return _ORTModuleFunction.apply(*self._convert_forward_input_to_list(*proc_inputs, **kwargs))

    def _convert_forward_input_to_list(self, *inputs, **kwargs):
        '''Creates forward `*inputs` list from user input and PyTorch initializers

        TODO: **kwargs is not supported

        ONNX Runtime forward requires an order list of:
            * User input: computed from forward InferenceSession
            * Initializers: computed from original PyTorch model parameters

        This codes assumes the exported model's inputs and initializers
            are the same as the original PyTorch model
        '''
        # List containing both user inputs and initializers, in this order
        result = []

        # Inputs
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            result.append(inputs[idx])

        # Initializers
        for idx, param in enumerate(self._original_module.named_parameters()):
            result.append(param[1])

        return result

    def _convert_dict_torch_to_numpy(self, tensor_dict):
        '''Convert `tensor_dict` PyTorch tensors to numpy tensors

        This is a ONNX Runtime requirement

        TODO: #UseIOBinding
        '''
        result = {}
        for k,v in tensor_dict.items():
            result.update({k : v.detach().cpu().numpy()})
        return result

    def _convert_forward_input_list_to_dict(self, *inputs):
        '''Convert forward `*inputs` list to dict

        TODO: Input gradient is being ignored for MVP
        '''
        # Dictionary containing both inputs and initializers
        result = {}

        # Inputs
        result_len = 0
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            result_len += 1
            result.update({input_data.name: inputs[idx]})

        # Initializers
        for param in self._original_module.named_parameters():
            result.update({param[0]: inputs[result_len]})
            result_len += 1

        return result

    def _convert_backward_input_list_to_dict(self, *inputs):
        '''Convert backward `*inputs` list to dict

        ONNX Runtime backward requires dict as input, which is composed of:
            * User input
                Although not necessary, all user inputs are used for simplicity
            * (Partial) Initializers
                    init_begin = len(user_input)
                    init_count = len(Pre-computed list of initializer)
            * Intermediate tensors
            * Gradient wrt outputs
        '''

        # Dictionary containing both inputs and initializers
        result = {}

        backward_user_input = self._onnx_graphs_info.backward_user_input_names
        backward_intializer = self._onnx_graphs_info.backward_intializer_names_as_input
        intermediate = self._onnx_graphs_info.intermediate_tensor_names
        backward_output_grad_names = self._onnx_graphs_info.backward_output_grad_names

        # Extract info about stashed input and grad output
        # Inputs
        inputs_pos = 0
        for idx, name in enumerate(backward_user_input):
            result.update({ name : inputs[idx]})
            inputs_pos += 1

        # Initializers
        for idx, name in enumerate(backward_intializer, inputs_pos):
            result.update({name: inputs[idx]})
            inputs_pos += 1

        # Intermediate
        for idx, name in enumerate(intermediate, inputs_pos):
            result.update({name: inputs[idx]})
            inputs_pos += 1

        # Grad outputs
        for idx, name in enumerate(backward_output_grad_names, inputs_pos):
            result.update({name: inputs[idx]})
            inputs_pos += 1

        return result

    def _run_forward_graph(self, inputs):
        '''Execute forward pass on ONNX Runtime

        Output order has to be specified to ONNX Runtime backend
            to distinguish intermediate from output tensors
        '''

        forward_output = self._forward_session.run([*self._onnx_graphs_info.user_output_names,
                                                    *self._onnx_graphs_info.intermediate_tensor_names], inputs)
        output = forward_output[:len(self._onnx_graphs_info.user_output_names)]
        intermediates = forward_output[len(self._onnx_graphs_info.user_output_names):]
        return output, intermediates

    def _run_backward_graph(self, *inputs, **kwargs):
        '''Execute backward pass on ONNX Runtime

        `*inputs` is converted from list to a list of detached numpy tensors before
        being fed to an ONNX Runtime InferenceSession

        TODO: **kwargs are not supported
        '''

        # Convert input to dict of torch tensors
        data = self._convert_backward_input_list_to_dict(*inputs)

        # Convert dict of torch tensors to dict of numpy arrays (ORT BE requirement)
        data = self._convert_dict_torch_to_numpy(data)
        return self._backward_session.run(self._onnx_graphs_info.initializer_grad_names_to_train, data)

    @staticmethod
    def _get_forward_graph(module, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        TODO: How to ingest **kwargs in proper order during export?
        '''
        # Export the model to memory
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        sample_inputs_copy = copy.deepcopy(inputs)

        # Ignore optional *inputs explicitly specified as None
        sig = signature(module.forward)
        all_input_names = sig.parameters.keys()
        input_names = [name for idx, name in enumerate(all_input_names) if inputs[idx] is not None]

        # TODO: Support contrib OPs support? user model has no hint
        # from onnxruntime.training import register_custom_ops_pytorch_exporter
        # register_custom_ops_pytorch_exporter.register_custom_op()

        # Export torch.nn.Module to ONNX
        torch.onnx.export(module,
                          tuple(sample_inputs_copy),
                          f,
                          input_names=input_names,
                          opset_version=ONNX_OPSET_VERSION,
                          do_constant_folding=False,
                          training=torch.onnx.TrainingMode.TRAINING)

        return onnx.load_model_from_string(f.getvalue())


    @staticmethod
    def _build_fw_bw_grad_graphs(forward_graph, config, initializer_names=[]):
        '''Adds gradient nodes on top of an existing ONNX graph (with training flag)'''
        if not config.initializer_names_to_train:
            if not initializer_names:
                initializer_names_to_train = []
                for initializer in forward_graph.graph.initializer:
                    initializer_names_to_train.append(initializer.name)
                config.initializer_names_to_train = initializer_names_to_train
            else:
                config.initializer_names_to_train = initializer_names

            # TODO: Add support to input with grad required
            config.input_names_require_grad = []
            # input_names_require_grad = []
            # input_names_require_grad.append('input.1')
            # config.input_names_require_grad = input_names_require_grad

        module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
        module_gradient_graph_builder.build_and_split(forward_graph.SerializeToString(), config)
        forward_model = onnx.load_model_from_string(module_gradient_graph_builder.get_forward_model())
        backward_model = onnx.load_model_from_string(module_gradient_graph_builder.get_backward_model())
        gradient_model = onnx.load_model_from_string(module_gradient_graph_builder.get_gradient_model())
        split_graphs_info = module_gradient_graph_builder.get_split_graphs_info()

        return gradient_model, forward_model, backward_model, split_graphs_info


    @staticmethod
    def _get_io_info_from_onnx_graph(model, graphs_info):
        type_map = {key: None for key in [
            *graphs_info.user_input_names,
            *graphs_info.initializer_names_to_train,
            *graphs_info.initializer_grad_names_to_train,
            *graphs_info.user_output_names,
            *graphs_info.intermediate_tensor_names,
            *graphs_info.user_output_grad_names
        ]}

        for input in model.graph.input:
            if input.name in type_map and type_map[input.name] is None:
                type_map[input.name] = input.type
            input_grad_name = input.name + '_grad'
            if input_grad_name in type_map and type_map[input_grad_name] is None:
                type_map[input_grad_name] = input.type

        for output in model.graph.output:
            if output.name in type_map and type_map[output.name] is None:
                type_map[output.name] = output.type
            output_grad_name = output.name + '_grad'
            if output_grad_name in type_map and type_map[output_grad_name] is None:
                type_map[output_grad_name] = output.type

        return type_map
