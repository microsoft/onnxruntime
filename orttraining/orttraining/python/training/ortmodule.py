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

from torch.utils.dlpack import from_dlpack

# Needed to re-implement PyTorch's cpu,cuda,to methods
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict

from onnxruntime.capi import _pybind_state as C
from . import _utils


ONNX_OPSET_VERSION = 12
__TEMP_ENABLE_METHOD_TIMING__ = False

# Needed to re-implement PyTorch's cpu,cuda,to methods
T = TypeVar('T', bound='Module')


def _get_device_index(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        device = torch.device(device)
    elif isinstance(device, int):
        return device
    return 0 if device.index is None else device.index


def _get_device_str(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        if device.find(':') == -1:
            device += ':' + str(torch.cuda.current_device())
    elif isinstance(device, int):
        device = 'cuda:' + str(device)
    elif isinstance(device, torch.device):
        if device.index is None:
            device = device.type + ':' + str(torch.cuda.current_device())
        else:
            device = device.type + ':' + str(device.index)
    else:
        raise ('Unsupported device type')
    return device


def _get_default_device_str(type):
    if type == 'cuda':
        return 'cuda:' + str(torch.cuda.current_device())
    else:
        return 'cpu'


def _create_iobinding(io_binding, inputs, model, device, output_buffers, len_user_outputs):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_input(value_info.name, inputs[idx].device.type,
                              _get_device_index(inputs[idx].device),
                              _utils.dtype_torch_to_numpy(inputs[idx].dtype),
                              list(inputs[idx].size()),
                              inputs[idx].data_ptr())

    for idx, value_info in enumerate(model.graph.output):
        name = value_info.name
        if idx < len_user_outputs:
            output_tensor = output_buffers[name]
            io_binding.bind_output(name, output_tensor.device.type,
                                   _get_device_index(device),
                                   _utils.dtype_torch_to_numpy(
                                       output_tensor.dtype),
                                   list(output_tensor.size()),
                                   output_tensor.data_ptr())
        else:
            io_binding.bind_output(name, device.type,
                                   device_id=_get_device_index(device))


def _onnx_value_info_to_buffer_tensor(value_info, device):
    '''Create a torch zeroed tensor with the same shape and type of `value_info`'''

    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    dtype = _utils.dtype_onnx_to_torch(value_info.type.tensor_type.elem_type)
    return torch.zeros(shape, device=device, dtype=dtype)

# TODO: PyTorch's to_dlpack() uses same config for both torch.bool and torch.uint8,
# and convert the config to torch.uint8 tensor duing from_dlpack(). So a boolean tensor
# from forward graph outputs will be converted to torch.uint8 tensor. When this tensor
# is feeded to backward graph as input, it will cause data type mismatch issue during
# inference session running. We cannot change the from_dlpack() in PyTorch side, so we
# have to handle this specially, which will introduce a cast here and there is data copied.
# Always cast from torch.uint8 to torch.bool is not logically right, we need to check the
# real data type of the inputs in the backeard graph, and perform the cast only necessary.


def _ort_output_to_torch_tensor(ort_output):
    tensor = from_dlpack(ort_output.to_dlpack())
    return tensor.to(torch.bool) if tensor.dtype == torch.uint8 else tensor


class ORTModule(torch.nn.Module):

    def __init__(self, module):
        assert isinstance(
            module, torch.nn.Module), "'module' mst be a torch.nn.Module"
        super(ORTModule, self).__init__()

        self._export_again = False
        # TODO: This is incorrect when different layers may be in different devices
        self._device = next(module.parameters()).device
        self._require_export = False

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        self._onnx_training = None

        # Related to training graph split/shape inference
        self._current_input_shape = None
        self._module_gradient_graph_builder = None

        # Forward pass
        self._onnx_forward = None
        self._forward_session = None
        self._forward_io_binding = None

        # Backward pass
        self._onnx_backward = None
        self._backward_session = None
        self._backward_io_binding = None

        self._onnx_gradient = None
        self._gradient_session = None
        self._gradient_io_binding = None
        self._output_buffers = {}

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

    def cpu(self: T) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if self._device.type != 'cpu':
            self._require_export = True
            self._device = torch.device('cpu')

        return super(ORTModule, self).cpu()

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if device is None:
            if _get_device_str(self._device) != _get_default_device_str('cuda'):
                self._require_export = True
                self._device = torch.device(_get_default_device_str('cuda'))
        elif _get_device_str(self._device) != _get_device_str(device):
            self._require_export = True
            self._device = torch.device(_get_device_str(device))

        return super(ORTModule, self).cuda(device)

    @overload
    def to(self: T, device: Optional[Union[int, torch.device]] = ...,
           dtype: Optional[Union[torch.dtype, str]] = ...,
           non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[torch.dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        '''Thin layer to capture device for ORTModule IO bindings'''

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device:
            device_str = _get_device_str(device)
            if _get_device_str(self._device) != device_str:
                self._require_export = True
                self._device = torch.device(device_str)
        return super(ORTModule, self).to(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, a full training graph is splitted in forward and backward graph which are used
        to instantiate ONNX Runtime InferenceSession`s
        '''
        if not self._onnx_gradient or self._require_export:
            self._require_export = False
            self._onnx_training = ORTModule._get_forward_graph(
                self._original_module, *inputs, **kwargs)

            if self._save_onnx:
                onnx.save(self._onnx_training,
                          self._save_onnx_prefix + '_full_training.onnx')

            # TODO: PyTorch exporter bug: changes the initializer order
            initializer_names = [p[0]
                                 for p in self._original_module.named_parameters()]

            # Build full training graph and split in forward/backward
            grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()
            grad_builder_config.initializer_names_to_train = initializer_names
            grad_builder_config.input_names_require_grad = []
            self._module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
            self._module_gradient_graph_builder.initialize(
                self._onnx_training.SerializeToString(), grad_builder_config)
            self._module_gradient_graph_builder.build()
            self._onnx_gradient = onnx.load_model_from_string(
                self._module_gradient_graph_builder.get_gradient_model())
            self._onnx_graphs_info = self._module_gradient_graph_builder.get_split_graphs_info()

            providers = None
            provider_options = None
            if self._device.type == 'cuda':
                # Configure the InferenceSessions to use the specific GPU on which the model is placed.
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                provider_options = [{"device_id": str(self._device.index)}]
                # Release CUDA cache used by PyTorch during exporter
                torch.cuda.empty_cache()
            elif self._device.type == 'cpu':
                providers = ["CPUExecutionProvider"]
                provider_options = [{}]

            self._gradient_session = onnxruntime.InferenceSession(
                self._onnx_gradient.SerializeToString(), providers=providers, provider_options=provider_options)
            # Use this global one for now, so forward and backward are sharing the same one.
            self._run_options = C.RunOptions()

            self._gradient_io_binding = self._gradient_session.io_binding()
            # Bind user outputs in old way, so create the buffer.
            for output in self._onnx_gradient.graph.output[:len(self._onnx_graphs_info.user_output_names)]:
                self._output_buffers[output.name] = _onnx_value_info_to_buffer_tensor(
                    output, str(self._device))

            if self._save_onnx:
                onnx.save(self._onnx_gradient,
                          self._save_onnx_prefix + '_gradient.onnx')

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

                # Use IO binding, bind inputs and non-user outputs.
                _create_iobinding(self._gradient_io_binding, inputs,
                                  self._onnx_gradient,
                                  self._device,
                                  self._output_buffers,
                                  len(self._onnx_graphs_info.user_output_names))

                # Run
                self._gradient_session.run_with_iobinding(
                    self._gradient_io_binding, self._run_options)

                # Return model output
                user_outputs = tuple(
                    self._output_buffers[name] for name in self._onnx_graphs_info.user_output_names)
                return user_outputs[0] if len(user_outputs) == 1 else user_outputs

            @staticmethod
            def backward(ctx, *grad_output):
                '''Performs backward pass based on grad wrt output and internal state

                Internal state is composed of:
                    * Tensor stashed (in a particular order) during forward:
                        * (partial) user input, (partial) initializers and intermediate tensors

                TODO: Input gradient is hard-coded to torch.tensor([1.])
                '''

                # Push user output grads to ONNX backend.
                grad_output_dict = dict(
                    zip(self._onnx_graphs_info.user_output_grad_names, grad_output))
                backward_grad_output = [grad_output_dict[name]
                                        for name in self._onnx_graphs_info.backward_output_grad_names]
                backward_grad_output_ortvalue = []
                for grad_output in backward_grad_output:
                    backward_grad_output_ortvalue.append(onnxruntime.OrtValue.ortvalue_from_data_ptr(list(grad_output.size()), _utils.dtype_torch_to_numpy(
                        grad_output.dtype), grad_output.device.type, _get_device_index(grad_output.device), grad_output.data_ptr()))

                # Run
                self._gradient_session.run_backward(
                    backward_grad_output_ortvalue)
                backward_outputs = self._gradient_io_binding.get_outputs(
                )[len(self._onnx_graphs_info.user_output_names):]

                # Return input and initializer gradients
                results = [torch.tensor(
                    [1])] * len(self._onnx_graphs_info.user_input_names)
                results += [_ort_output_to_torch_tensor(backward_output)
                            for backward_output in backward_outputs[:len(self._onnx_graphs_info.initializer_grad_names_to_train)]]
                return tuple(results)

        return _ORTModuleFunction.apply(*self._convert_forward_input_to_list(self._original_module, *inputs, **kwargs))

    @_utils.timeit(enabled=__TEMP_ENABLE_METHOD_TIMING__)
    def _convert_forward_input_to_list(self, module, *inputs, **kwargs):
        '''Creates forward `*inputs` list from user input and PyTorch initializers

        TODO: **kwargs is not supported
        TODO: How IO binding model inputs and outputs affects initializer copies?

        ONNX Runtime forward requires an order list of:
            * User input: computed from forward InferenceSession
            * Initializers: computed from original PyTorch model parameters

        This codes assumes the exported model's inputs and initializers
            are the same as the original PyTorch model
        '''
        # User inputs
        _, input_tensors = ORTModule._extract_user_inputs(
            module, *inputs, **kwargs)
        result = [tensor for tensor in input_tensors if tensor is not None]

        # Initializers
        for param in self._original_module.named_parameters():
            result.append(param[1])

        return result

    @_utils.timeit(enabled=__TEMP_ENABLE_METHOD_TIMING__)
    def _convert_forward_input_list_to_dict(self, *inputs):
        '''Convert forward `*inputs` list to dict

        TODO: Input gradient is being ignored for MVP
        '''
        # Dictionary containing both inputs and initializers
        forward_input_names = [*self._onnx_graphs_info.user_input_names,
                               *self._onnx_graphs_info.initializer_names_to_train]
        return dict(zip(forward_input_names, inputs))

    @_utils.timeit(enabled=__TEMP_ENABLE_METHOD_TIMING__)
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
            result.update({name: inputs[idx]})
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

    @staticmethod
    def _get_forward_graph(module, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        TODO: How to ingest **kwargs in proper order during export?
        '''
        # Export the model to memory
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # sample_inputs_copy = copy.deepcopy(inputs)

        # Ignore optional *inputs explicitly specified as None
        '''
        sig = signature(module.forward)
        all_input_names = sig.parameters.keys()
        input_names = []
        dynamic_axes = {}
        for input_idx, name in enumerate(all_input_names):
            if input_idx < len(inputs) and inputs[input_idx] is not None:
                input_names.append(name)
                dynamic_axes[name] = {}
                 for dim_idx in range(len(inputs[input_idx].shape)):
                    dynamic_axes[name].update({dim_idx : f'input{input_idx}_dim{dim_idx}'})
        '''

        input_names, input_tensors = ORTModule._extract_user_inputs(
            module, *inputs, **kwargs)

        # TODO: Support contrib OPs support? user model has no hint
        # from onnxruntime.training import register_custom_ops_pytorch_exporter
        # register_custom_ops_pytorch_exporter.register_custom_op()

        # Export torch.nn.Module to ONNX
        torch.onnx.export(module,
                          tuple(input_tensors),
                          f,
                          input_names=input_names,
                          opset_version=ONNX_OPSET_VERSION,
                          do_constant_folding=False,
                          training=torch.onnx.TrainingMode.TRAINING,)
        # dynamic_axes=dynamic_axes)

        return onnx.load_model_from_string(f.getvalue())

    @staticmethod
    def _extract_user_inputs(module, *inputs, **kwargs):
        sig = signature(module.forward)
        all_input_names = sig.parameters.keys()
        input_names = []
        input_tensors = []
        for idx, name in enumerate(all_input_names):
            tensor = None
            if idx < len(inputs) and inputs[idx] is not None:
                input_names.append(name)
                tensor = inputs[idx]
            if name in kwargs:
                if tensor is None:
                    input_names.append(name)
                tensor = kwargs[name]
            input_tensors.append(tensor)

        return input_names, input_tensors
