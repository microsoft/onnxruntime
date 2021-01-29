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

def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_input(value_info.name, inputs[idx].device.type,
                              _get_device_index(inputs[idx].device),
                              _utils.dtype_torch_to_numpy(inputs[idx].dtype),
                              list(inputs[idx].size()),
                              inputs[idx].data_ptr())

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type,
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

def _extract_input_information(module, *inputs, **kwargs):
    '''Returns the all input names, dynamic_axes information and input names that require gradient'''

    # Ignore optional *inputs explicitly specified as None
    sig = signature(module.forward)
    all_input_names = sig.parameters.keys()
    input_names = []
    dynamic_axes = {}
    input_names_require_grad = []
    for input_idx, name in enumerate(all_input_names):
        if input_idx < len(inputs) and inputs[input_idx] is not None:
            if inputs[input_idx].requires_grad:
                # input_names_require_grad holds all input tensors that have requires_grad
                input_names_require_grad.append(name)

            input_names.append(name)
            dynamic_axes[name] = {}
            for dim_idx in range(len(inputs[input_idx].shape)):
                dynamic_axes[name].update({dim_idx : f'input{input_idx}_dim{dim_idx}'})

    return input_names, dynamic_axes, input_names_require_grad

class ORTModule(torch.nn.Module):

    def __init__(self, module):
        assert isinstance(module, torch.nn.Module), "'module' mst be a torch.nn.Module"
        super(ORTModule, self).__init__()

        self._export_again = False
        # TODO: This is incorrect when different layers may be in different devices
        self._device = next(module.parameters()).device
        self._device_changed = False

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        self._onnx_training = None

        # Related to training graph split/shape inference
        self._current_input_shape = None
        self._module_gradient_graph_builder = None
        self._input_names_require_grad = None

        # Forward pass
        self._onnx_forward = None
        self._forward_session = None
        self._forward_io_binding = None

        # Backward pass
        self._onnx_backward = None
        self._backward_session = None
        self._backward_io_binding = None

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

    def _initialize_module_gradient_graph_builder(self):

        # TODO: PyTorch exporter bug: changes the initializer order
        initializer_names = [p[0] for p in self._original_module.named_parameters()]

        # Build full training graph and split in forward/backward
        grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()
        grad_builder_config.initializer_names_to_train = initializer_names
        grad_builder_config.input_names_require_grad = self._input_names_require_grad
        self._module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
        self._module_gradient_graph_builder.initialize(self._onnx_training.SerializeToString(), grad_builder_config)

    def _build_training_graph(self, *inputs, **kwargs):
        input_names, dynamic_axes, self._input_names_require_grad = \
                _extract_input_information(self._original_module, *inputs, **kwargs)
        self._onnx_training = self._get_forward_graph(input_names, dynamic_axes, *inputs, **kwargs)
        if self._save_onnx:
            onnx.save(self._onnx_training, self._save_onnx_prefix + '_full_training.onnx')

        self._initialize_module_gradient_graph_builder()

    def _create_training_session(self):
        providers = None
        provider_options = None
        if self._device.type == 'cuda':
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": str(self._device.index)}, {}]
        elif self._device.type == 'cpu':
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        self._forward_session = onnxruntime.InferenceSession(
            self._onnx_forward.SerializeToString(), providers=providers, provider_options=provider_options)
        self._backward_session = onnxruntime.InferenceSession(
            self._onnx_backward.SerializeToString(), providers=providers, provider_options=provider_options)

        # IO binding
        # TODO: we should try to reuse the output buffers as some of the output tensors are same sizes, expecially the backward graph outputs.
        self._forward_io_binding = self._forward_session.io_binding()
        self._backward_io_binding = self._backward_session.io_binding()

    def _split_training_graph(self, *inputs, **kwargs):
        # Perform shape inference and re-split forward/backward graph for batches with different shapes
        self._module_gradient_graph_builder.build_and_split(self._current_input_shape)
        self._onnx_forward = onnx.load_model_from_string(self._module_gradient_graph_builder.get_forward_model())
        self._onnx_backward = onnx.load_model_from_string(self._module_gradient_graph_builder.get_backward_model())
        self._onnx_graphs_info = self._module_gradient_graph_builder.get_split_graphs_info()
        self._create_training_session()

        if self._save_onnx:
            onnx.save(self._onnx_forward, self._save_onnx_prefix + '_forward.onnx')
            onnx.save(self._onnx_backward, self._save_onnx_prefix + '_backward.onnx')

    def cpu(self: T) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if self._device.type != 'cpu':
            self._device_changed = True
            self._device = torch.device('cpu')

        return super(ORTModule, self).cpu()

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if device is None:
            if _get_device_str(self._device) != _get_default_device_str('cuda'):
                self._device_changed = True
                self._device = torch.device(_get_default_device_str('cuda'))
        elif _get_device_str(self._device) != _get_device_str(device):
            self._device_changed = True
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
                self._device_changed = True
                self._device = torch.device(device_str)
        return super(ORTModule, self).to(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, a full training graph is splitted in forward and backward graph which are used
        to instantiate ONNX Runtime InferenceSession`s
        '''

        # Exporting module to ONNX for the first time
        if not self._onnx_training:
            self._build_training_graph(*inputs, **kwargs)

        _, _, input_names_require_grad = _extract_input_information(self._original_module, *inputs, **kwargs)
        # If inputs requiring gradient change from one call to forward to the next, the module_gradient_graph_builder
        # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
        if input_names_require_grad != self._input_names_require_grad:
            self._input_names_require_grad = input_names_require_grad
            self._initialize_module_gradient_graph_builder()

        new_input_shape = [list(input.size()) for input in inputs if input is not None]
        if self._current_input_shape is None or self._current_input_shape != new_input_shape:
            self._current_input_shape = new_input_shape
            self._split_training_graph(*inputs, **kwargs)
        elif self._device_changed:
            self._create_training_session()
            self._device_changed = False

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

                # Use IO binding
                _create_iobinding(self._forward_io_binding, inputs,
                                  self._onnx_forward,
                                  self._device)

                # Run
                self._forward_session.run_with_iobinding(self._forward_io_binding)
                forward_outputs = self._forward_io_binding.get_outputs()

                # Stash tensors needed by backward
                forward_input_dict = self._convert_forward_input_list_to_dict(*inputs)
                ctx_inputs = tuple(forward_input_dict[name] \
                    for name in self._onnx_graphs_info.backward_user_input_names)
                ctx_initializers = tuple(forward_input_dict[name] \
                    for name in self._onnx_graphs_info.backward_intializer_names_as_input)
                ctx_intermediates = tuple(_ort_output_to_torch_tensor(forward_output) \
                    for forward_output in forward_outputs[len(self._onnx_graphs_info.user_output_names):])
                ctx.save_for_backward(*[*ctx_inputs, *ctx_initializers, *ctx_intermediates])

                # Return model output
                user_outputs = tuple(_ort_output_to_torch_tensor(forward_output) \
                    for forward_output in forward_outputs[:len(self._onnx_graphs_info.user_output_names)])
                return user_outputs[0] if len(user_outputs) == 1 else user_outputs

            @staticmethod
            def backward(ctx, *grad_output):
                '''Performs backward pass based on grad wrt output and internal state

                Internal state is composed of:
                    * Tensor stashed (in a particular order) during forward:
                        * (partial) user input, (partial) initializers and intermediate tensors

                TODO: Input gradient is hard-coded to torch.tensor([1.])
                '''

                # Use IO binding
                grad_output_dict = dict(zip(self._onnx_graphs_info.user_output_grad_names, grad_output))
                backward_grad_output = tuple(grad_output_dict[name] for name in self._onnx_graphs_info.backward_output_grad_names)
                _create_iobinding(self._backward_io_binding, [*ctx.saved_tensors, *backward_grad_output],
                                   self._onnx_backward,
                                   self._device)

                # Run
                self._backward_session.run_with_iobinding(self._backward_io_binding)
                backward_outputs = self._backward_io_binding.get_outputs()

                # Return input and initializer gradients
                num_initializers = len(self._onnx_graphs_info.initializer_grad_names_to_train)
                results = []
                for input_name in self._onnx_graphs_info.user_input_names:
                    try:
                        # Append to the results the backward output for each input that required grad
                        results.append(_ort_output_to_torch_tensor(
                            backward_outputs[num_initializers + self._input_names_require_grad.index(input_name)]))
                    except ValueError:
                        # Append None to results for each input that did not require grad
                        results.append(None)
                # Append backward ouput for all trained initializers
                results += [_ort_output_to_torch_tensor(backward_output)
                            for backward_output in backward_outputs[:num_initializers]]
                return tuple(results)

        proc_inputs = [data for data in inputs if data is not None]
        return _ORTModuleFunction.apply(*self._convert_forward_input_to_list(*proc_inputs, **kwargs))

    @_utils.timeit(enabled=__TEMP_ENABLE_METHOD_TIMING__)
    def _convert_forward_input_to_list(self, *inputs, **kwargs):
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
        result = list(inputs[:len(self._onnx_graphs_info.user_input_names)])

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

    def _get_forward_graph(self, input_names, dynamic_axes, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        TODO: How to ingest **kwargs in proper order during export?
        '''
        # Export the model to memory
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        sample_inputs_copy = []
        for model_input in inputs:
            sample_inputs_copy.append(model_input.data if isinstance(model_input, torch.Tensor) else model_input)
        sample_inputs_copy = copy.deepcopy(tuple(sample_inputs_copy))

        # TODO: Support contrib OPs support? user model has no hint
        # from onnxruntime.training import register_custom_ops_pytorch_exporter
        # register_custom_ops_pytorch_exporter.register_custom_op()

        with torch.no_grad():
            # Export torch.nn.Module to ONNX
            torch.onnx.export(self._original_module,
                              sample_inputs_copy,
                              f,
                              input_names=input_names,
                              opset_version=ONNX_OPSET_VERSION,
                              do_constant_folding=False,
                              training=torch.onnx.TrainingMode.TRAINING,
                              dynamic_axes=dynamic_axes)

        return onnx.load_model_from_string(f.getvalue())
