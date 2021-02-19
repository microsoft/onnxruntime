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
from torch.utils.cpp_extension import load_inline
from collections import abc, OrderedDict

# Needed to re-implement PyTorch's cpu,cuda,to methods
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict

from onnxruntime.capi import _pybind_state as C
from onnxruntime.training import register_custom_ops_pytorch_exporter
from . import _utils


ONNX_OPSET_VERSION = 12
__TEMP_ENABLE_METHOD_TIMING__ = False

# Needed to re-implement PyTorch's cpu,cuda,to methods
T = TypeVar('T', bound='Module')


def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_input(value_info.name, inputs[idx].device.type,
                              _utils.get_device_index(inputs[idx].device),
                              _utils.dtype_torch_to_numpy(inputs[idx].dtype),
                              list(inputs[idx].size()),
                              inputs[idx].data_ptr())

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type,
                               device_id=_utils.get_device_index(device))

def _deepcopy_model_input(*inputs, **kwargs):
    sample_inputs_copy = []
    for model_input in inputs:
        sample_inputs_copy.append(model_input.data if isinstance(model_input, torch.Tensor) else model_input)
    sample_inputs_copy = copy.deepcopy(tuple(sample_inputs_copy))

    sample_kwargs_copy = {}
    for name, model_input in kwargs.items():
        sample_kwargs_copy[name] = model_input.data if isinstance(model_input, torch.Tensor) else model_input
    sample_kwargs_copy = copy.deepcopy(sample_kwargs_copy)

    return sample_inputs_copy, sample_kwargs_copy

def _onnx_value_info_to_buffer_tensor(value_info, device):
    '''Create a torch zeroed tensor with the same shape and type of `value_info`'''

    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    dtype = _utils.dtype_onnx_to_torch(value_info.type.tensor_type.elem_type)
    return torch.zeros(shape, device=device, dtype=dtype)

def _parse_inputs_for_onnx_export(all_input_names, onnx_graph, *inputs, **kwargs):
    # Ignore optional inputs explicitly specified as None
    # ONNX exporter may remove unused inputs
    onnx_graph_input_names = []
    if onnx_graph is not None:
        onnx_graph_input_names = set([inp.name for inp in onnx_graph.graph.input])

    input_names = []
    dynamic_axes = {}
    input_names_require_grad = []
    input_shape = []

    for input_idx, name in enumerate(all_input_names):
        inp = None
        if input_idx < len(inputs) and inputs[input_idx] is not None:
            inp = inputs[input_idx]
        elif name in kwargs and kwargs[name] is not None:
            inp = kwargs[name]
        if inp is not None and (onnx_graph is None or name in onnx_graph_input_names):
            if inp.requires_grad:
                # input_names_require_grad holds all input tensors that have requires_grad
                input_names_require_grad.append(name)

            input_names.append(name)
            dynamic_axes[name] = {}
            for dim_idx in range(len(inp.shape)):
                dynamic_axes[name].update({dim_idx : f'input{input_idx}_dim{dim_idx}'})

            input_shape.append(list(inp.size()))
    return input_names, dynamic_axes, input_names_require_grad, input_shape

def _process_output_name(output_name, output_index = None):
    processed_name = output_name.replace('*', '**')
    return processed_name if output_index is None else processed_name + '*{}'.format(output_index)

def _unprocess_output_name(processed_name):
    index_appended = processed_name.count('*') % 2 == 1
    output_index = None
    if index_appended:
        output_index = int(processed_name[processed_name.rfind('*')+1:])
        processed_name = processed_name[:processed_name.rfind('*')]
    return processed_name.replace('**', '*'), output_index

def _parse_outputs_for_onnx_export(module, inputs, kwargs):

    def _create_output_dim_names_from_mapping(output):
        output_names, dynamic_axes, use_derived_module = [], {}, False
        for name, value in output.items():
            if isinstance(value, torch.Tensor):
                processed_name = _process_output_name(name)
                output_names.append(processed_name)
                dynamic_axes[processed_name] = {}
                for dim_idx in range(len(value.shape)):
                    dynamic_axes[processed_name].update({dim_idx: '{}_dim{}'.format(processed_name, dim_idx)})
            elif isinstance(value, abc.Sequence):
                use_derived_module = True
                for i, sequence_value in enumerate(value):
                    if not isinstance(sequence_value, torch.Tensor):
                        raise TypeError('ORTModule does not support the following model output type {} \
                            within a Sequence within a Mapping'.format(type(output)))
                    processed_name = _process_output_name(name, i)
                    dynamic_axes[processed_name] = {}
                    output_names.append(processed_name)
                    for dim_idx in range(len(sequence_value.shape)):
                        dynamic_axes[processed_name].update({dim_idx: '{}_dim{}'.format(processed_name, dim_idx)})
            else:
                raise TypeError('ORTModule does not support the following model output type {} within a Mapping'.format(type(value)))

        return output_names, dynamic_axes, use_derived_module

    def _create_output_dim_names(output, output_idx):
        output_names, dynamic_axes, use_derived_module = [], {}, False
        name = 'output{}'.format(output_idx)
        if isinstance(output, torch.Tensor):
            processed_name = _process_output_name(name)
            output_names.append(processed_name)
            dynamic_axes[processed_name] = {}
            for dim_idx in range(len(output.shape)):
                dynamic_axes[processed_name].update({dim_idx : '{}_dim{}'.format(processed_name, dim_idx)})
        elif isinstance(output, abc.Sequence):
            use_derived_module = True
            for i, sequence_value in enumerate(output):
                if not isinstance(sequence_value, torch.Tensor):
                    raise TypeError('ORTModule does not support the following model output type {} \
                        within a Sequence within a Sequence'.format(type(output)))
                processed_name = _process_output_name(name, i)
                dynamic_axes[processed_name] = {}
                output_names.append(processed_name)
                for dim_idx in range(len(sequence_value.shape)):
                    dynamic_axes[processed_name].update({dim_idx: '{}_dim{}'.format(processed_name, dim_idx)})
        else:
            raise TypeError('ORTModule does not support the following model output type {} within a Sequence'.format(type(output)))
        return output_names, dynamic_axes, use_derived_module

    #   Do an inference to grab outputs
    is_train_mode = module.training
    module.eval()
    output_names = []
    output_dynamic_axes = {}
    sample_output_type = None
    use_derived_module = False
    with torch.no_grad():
        # Deepcopy inputs, since input values may change after model run.
        sample_inputs_copy, sample_kwargs_copy = _deepcopy_model_input(*inputs, **kwargs)
        try:
            # Deepcopy model, in case model is stateful and changes after model run.
            model_copy = copy.deepcopy(module)
        except Exception:
            model_copy = module
            warnings.warn("This model cannot be deep copied (or pickled), which is a required step for stateful models to be properly exported to ONNX."
                            " Compute will continue, but unexpected results may occur!")

        sample_outputs = model_copy(*sample_inputs_copy, **sample_kwargs_copy)
        sample_output_type = type(sample_outputs)
        if isinstance(sample_outputs, torch.Tensor):
            output_names, output_dynamic_axes, _ = _create_output_dim_names(sample_outputs, 0)
        elif isinstance(sample_outputs, abc.Mapping):
            output_names, output_dynamic_axes, use_derived_module = _create_output_dim_names_from_mapping(sample_outputs)
        elif isinstance(sample_outputs, abc.Sequence):
            for idx, out in enumerate(sample_outputs):
                tmp_output_names, tmp_output_dynamic_axes, tmp_use_derived_module = _create_output_dim_names(out, idx)
                use_derived_module = use_derived_module or tmp_use_derived_module
                output_names += tmp_output_names
                output_dynamic_axes.update(tmp_output_dynamic_axes)
        else:
            raise TypeError('ORTModule does not support the following model output type {}'.format(type(sample_outputs)))
    if is_train_mode:
        module.train()
    return output_names, output_dynamic_axes, sample_output_type, use_derived_module

def _populate_user_output(user_output_type, user_output_names, user_outputs):
    def _key_value_pairs_from_output(output_names, outputs):
        key_value_pairs = []
        for i, name in enumerate(user_output_names):
            output_name, output_index = _unprocess_output_name(name)
            if output_index is None:
                key_value_pairs.append((output_name, user_outputs[i]))
            elif output_index == 0:
                key_value_pairs.append((output_name, [user_outputs[i]]))
            else:
                key_value_pairs[-1][1].append(user_outputs[i])
        for i, _ in enumerate(key_value_pairs):
            if isinstance(key_value_pairs[i][1], abc.Sequence):
                key_value_pairs[i] = (key_value_pairs[i][0], tuple(key_value_pairs[i][1]))
        return key_value_pairs

    key_value_pairs = _key_value_pairs_from_output(user_output_names, user_outputs)
    if issubclass(user_output_type, Mapping):
        return user_output_type(key_value_pairs)
    elif issubclass(user_output_type, tuple):
        try:
            # Try constructing the user named tuple from the output tuple
            return user_output_type(*[user_output for _, user_output in key_value_pairs])
        except TypeError:
            # The expected output type is not a namedtuple, but is a regular tuple type
            pass

    return key_value_pairs[0][1] if len(key_value_pairs) == 1 \
        else tuple(user_output for _, user_output in key_value_pairs)

def _transform_to_flat_structure(output):
    def _transform_output_from_mapping(output_dict):
        transformed_output = OrderedDict()
        for key, value in output_dict.items():
            if isinstance(value, torch.Tensor):
                processed_key = _process_output_name(key)
                transformed_output[processed_key] = value
            elif isinstance(value, abc.Sequence):
                for i, sequence_value in enumerate(value):
                    if not isinstance(sequence_value, torch.Tensor):
                        raise TypeError('ORTModule does not support the following output type {} \
                            within a Sequence within a Mapping'.format(type(sequence_value)))
                    processed_key = _process_output_name(key, i)
                    transformed_output[processed_key] = sequence_value
            else:
                raise TypeError('ORTModule does not support the following output type {} within a Mapping'.format(type(value)))
        return transformed_output

    def _transform_output_from_sequence(output_sequence):
        transformed_output = []
        for value in output_sequence:
            if isinstance(value, torch.Tensor):
                transformed_output.append(value)
            elif isinstance(value, abc.Sequence):
                for sequence_value in value:
                    if not isinstance(sequence_value, torch.Tensor):
                        raise TypeError('ORTModule does not support the following output type {} \
                            within a Sequence within a Sequence'.format(type(sequence_value)))
                    transformed_output.append(sequence_value)
            else:
                raise TypeError('ORTModule does not support the following output type {} within a Sequence'.format(type(value)))
        return tuple(transformed_output)

    output_structure = None
    if isinstance(output, abc.Mapping):
        output_structure = _transform_output_from_mapping(output)
    elif isinstance(output, abc.Sequence):
        output_structure = _transform_output_from_sequence(output)
    return output_structure

class _DerivedModule(torch.nn.Module):
    def __init__(self, module):
        super(_DerivedModule, self).__init__()
        self._base_module = module

    def forward(self, *args, **kwargs):
        return _transform_to_flat_structure(self._base_module(*args, **kwargs))

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

def _load_torch_allocator_cpp_extension():
    torch_cuda_allocator_addresses_cpp_source = """
    #include <torch/extension.h>
    #include <c10/cuda/CUDACachingAllocator.h>
    size_t cuda_caching_allocator_raw_alloc_address() {
        return reinterpret_cast<size_t>(&c10::cuda::CUDACachingAllocator::raw_alloc);
    }
    size_t cuda_caching_allocator_raw_delete_address() {
        return reinterpret_cast<size_t>(&c10::cuda::CUDACachingAllocator::raw_delete);
    }
    """

    return load_inline(name='inline_extension', cpp_sources=[torch_cuda_allocator_addresses_cpp_source],
                        functions=['cuda_caching_allocator_raw_alloc_address', 'cuda_caching_allocator_raw_delete_address'],
                        verbose=True, with_cuda=True)

class ORTModule(torch.nn.Module):

    def __init__(self, module):
        assert isinstance(module, torch.nn.Module), "'module' must be a torch.nn.Module"
        super(ORTModule, self).__init__()

        # Support contrib OPs
        register_custom_ops_pytorch_exporter.register_custom_op()

        # TODO: Single device support for now
        self._device = _utils.get_device_from_module(module)
        self._device_changed = False

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        sig = signature(self._original_module.forward)
        self._original_module_input_names = sig.parameters.keys()
        self._derived_module = module
        self._onnx_inference = None
        self._is_training = True

        # Related to training graph shape inference
        self._current_input_shape = None
        self._module_gradient_graph_builder = None
        self._input_names_require_grad = None
        self._original_module_output_type = None

        # Training model
        self._onnx_training = None
        self._training_session = None
        self._training_io_binding = None
        self._run_options = None

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

        # CPP extension to get torch CUDA allocator's alloc and free function addresses
        self._use_external_cuda_allocator = True
        if self._use_external_cuda_allocator:
            self._torch_cuda_allocator = _load_torch_allocator_cpp_extension()
            self._torch_alloc = self._torch_cuda_allocator.cuda_caching_allocator_raw_alloc_address()
            self._torch_free = self._torch_cuda_allocator.cuda_caching_allocator_raw_delete_address()

    def _initialize_module_gradient_graph_builder(self):
        # TODO: PyTorch exporter bug: changes the initializer order
        initializer_names = [p[0] for p in self._original_module.named_parameters()]

        # Build full training graph
        grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()
        grad_builder_config.initializer_names_to_train = initializer_names
        grad_builder_config.input_names_require_grad = self._input_names_require_grad
        self._module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
        self._module_gradient_graph_builder.initialize(self._onnx_inference.SerializeToString(), grad_builder_config)

    def _get_inference_graph_and_init_gradient_graph_builder(self, *inputs, **kwargs):
        self._onnx_inference = self._get_inference_graph(*inputs, **kwargs)

        if self._save_onnx:
            onnx.save(self._onnx_inference, self._save_onnx_prefix + '_inference.onnx')

        self._initialize_module_gradient_graph_builder()

    def _create_training_session(self):
        providers = None
        provider_options = None
        if self._device.type == 'cuda':
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._use_external_cuda_allocator:
                provider_options = [{"device_id": str(self._device.index), "cuda_external_alloc": str(self._torch_alloc), "cuda_external_free": str(self._torch_free)}, {}]
            else:
                provider_options = [{"device_id": str(self._device.index)}, {}]
        elif self._device.type == 'cpu':
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.use_deterministic_compute = False
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = 2

        self._training_session = onnxruntime.InferenceSession(
            self._onnx_training.SerializeToString(), session_options, providers=providers, provider_options=provider_options)
        
        # Use this global run_options for now
        self._run_options = C.RunOptions()

        # IO binding
        # TODO: we should try to reuse the output buffers as some of the output tensors are same sizes, expecially the backward graph outputs.
        self._training_io_binding = self._training_session.io_binding()

    def _build_training_graph(self, *inputs, **kwargs):
        self._module_gradient_graph_builder.build(self._current_input_shape)
        self._onnx_training = onnx.load_model_from_string(self._module_gradient_graph_builder.get_training_model())
        self._onnx_graphs_info = self._module_gradient_graph_builder.get_training_graph_info()

        if self._save_onnx:
            onnx.save(self._onnx_training, self._save_onnx_prefix + '_training.onnx')

    def cpu(self: T) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if not self._device or self._device.type != 'cpu':
            self._device_changed = True
            self._device = torch.device('cpu')

        return super(ORTModule, self).cpu()

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''

        if device is None:
            if self._device and _utils.get_device_str(self._device) != _utils.get_default_device_str('cuda'):
                self._device_changed = True
                self._device = torch.device(_utils.get_default_device_str('cuda'))
        elif not self._device or _utils.get_device_str(self._device) != _utils.get_device_str(device):
            self._device_changed = True
            self._device = torch.device(_utils.get_device_str(device))

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
            try:
                device_str = _utils.get_device_str(device)
                if _utils.get_device_str(self._device) != device_str:
                    self._device_changed = True
                    self._device = torch.device(device_str)
            except RuntimeError:
                self._device_changed = True
                self._device = torch.device(device_str)

        return super(ORTModule, self).to(*args, **kwargs)

    def eval(self: T) -> T:
        self._is_training = False
        self._original_module.eval()

    def train(self: T, mode: bool = True) -> T:
        self._is_training = mode
        self._original_module.train(mode)

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, we build a full training graph with module_gradient_graph_builder. 
        Finally, we instantiate the ONNX Runtime InferenceSession.
        '''
        # TODO: using pytorch for evaluation for now. We will use ORT for evaluation latter. 
        if not self._is_training:
            return self._original_module(*inputs, **kwargs)

        # Exporting module to ONNX for the first time
        if not self._onnx_training:
            if not self._device:
                self._device = _utils.get_device_from_input_args_kwargs(self._original_module, *inputs, **kwargs)
                if not self._device:
                    raise RuntimeError('A device must be specified in the model or data!')
            self._get_inference_graph_and_init_gradient_graph_builder(*inputs, **kwargs)

        _, _, input_names_require_grad, new_input_shape = _parse_inputs_for_onnx_export(self._original_module_input_names, self._onnx_inference, *inputs, **kwargs)
        # If inputs requiring gradient change from one call to forward to the next, the module_gradient_graph_builder
        # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
        if input_names_require_grad != self._input_names_require_grad:
            self._input_names_require_grad = input_names_require_grad
            self._initialize_module_gradient_graph_builder()

        if self._current_input_shape is None or self._current_input_shape != new_input_shape:
            self._current_input_shape = new_input_shape
            self._build_training_graph()
            self._create_training_session()
        # TODO: disabled for now, since it caused a bug in NVBert fp32 run
        # When creating a new InferenceSession, there is a bug for destructing the original InferenceSession 
        # elif self._device_changed:
        #     self._create_training_session()
        #     self._device_changed = False

        # Use a custom torch.autograd.Function to associate self.backward_graph as the
        # gradient implementation for self.forward_graph.
        class _ORTModuleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                '''Performs forward pass based on user input and PyTorch initializer

                Autograd Function's apply() doesn't support keyword arguments,
                so `*inputs` has all the arguments - keyword arguments converted
                to positional by the caller.

                Module outputs are returned to the user
                '''

                # Use IO binding
                _create_iobinding(self._training_io_binding, inputs, self._onnx_training, self._device)

                # Run and return module outputs.
                user_outputs = tuple(_ort_output_to_torch_tensor(forward_output) \
                    for forward_output in self._training_session.run_forward(self._training_io_binding, self._run_options))
                return user_outputs[0] if len(user_outputs) == 1 else user_outputs

            @staticmethod
            def backward(ctx, *grad_output):
                '''Performs backward pass based on grad wrt module output
                '''

                # Use IO binding
                # Push user output grads to ONNX backend.
                backward_grad_output_ortvalue = []
                for grad_output in grad_output[:len(self._onnx_graphs_info.backward_output_grad_names)]:
                    backward_grad_output_ortvalue.append(onnxruntime.OrtValue.ortvalue_from_data_ptr(list(grad_output.size()), _utils.dtype_torch_to_numpy(
                        grad_output.dtype), grad_output.device.type, _utils.get_device_index(grad_output.device), grad_output.data_ptr()))

                # Run and get results
                self._training_session.run_backward(backward_grad_output_ortvalue)
                backward_outputs = self._training_io_binding.get_outputs()

                # Return input and initializer gradients
                num_user_input_grads = len(self._input_names_require_grad)

                results = []
                for input_name in self._onnx_graphs_info.user_input_names:
                    try:
                        # Append to the results the backward output for each input that required grad
                        results.append(_ort_output_to_torch_tensor(
                            backward_outputs[self._input_names_require_grad.index(input_name)]))
                    except ValueError:
                        # input_name is not found in the self._input_names_require_grad list
                        # Append None to results for each input that did not require grad
                        results.append(None)
                # Append gradients of initializer to results
                results += [_ort_output_to_torch_tensor(backward_output) 
                            for backward_output in backward_outputs[num_user_input_grads:]]
                return tuple(results)

        return _populate_user_output(self._original_module_output_type, self._onnx_graphs_info.user_output_names,
            _ORTModuleFunction.apply(*self._convert_training_graph_input_to_list(*inputs, **kwargs)))

    @_utils.timeit(enabled=__TEMP_ENABLE_METHOD_TIMING__)
    def _convert_training_graph_input_to_list(self, *inputs, **kwargs):
        '''Creates forward `*inputs` list from user input and PyTorch initializers

        TODO: How IO binding model inputs and outputs affects initializer copies?

        ONNX Runtime forward requires an order list of:
            * User input: computed from forward InferenceSession
            * Initializers: computed from original PyTorch model parameters
        '''
        # User inputs
        result = []
        for input_idx, name in enumerate(self._original_module_input_names):
            inp = None
            if input_idx < len(inputs) and inputs[input_idx] is not None:
                inp = inputs[input_idx]
            elif name in kwargs and kwargs[name] is not None:
                inp = kwargs[name]
            if inp is not None and name in self._onnx_graphs_info.user_input_names:
                result.append(inp)

        # Initializers
        for param in self._original_module.named_parameters():
            result.append(param[1])

        return result

    def _get_inference_graph(self, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        TODO: How to ingest **kwargs in proper order during export?
        '''

        # Setup dynamic axes for onnx model
        input_names, dynamic_axes, self._input_names_require_grad, _ = _parse_inputs_for_onnx_export(self._original_module_input_names, None, *inputs, **kwargs)
        output_names, output_dynamic_axes, self._original_module_output_type, use_derived_module = _parse_outputs_for_onnx_export(self._original_module, inputs, kwargs)
        dynamic_axes.update(output_dynamic_axes)

        if use_derived_module:
            self._original_module = _DerivedModule(self._original_module)

        # Export torch.nn.Module to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        sample_inputs_copy, sample_kwargs_copy = _deepcopy_model_input(*inputs, **kwargs)

        try:
            with torch.no_grad():
                torch.onnx.export(self._original_module,
                                sample_inputs_copy + (sample_kwargs_copy, ),
                                f,
                                input_names=input_names,
                                output_names=output_names,
                                opset_version=ONNX_OPSET_VERSION,
                                do_constant_folding=False,
                                training=torch.onnx.TrainingMode.TRAINING,
                                dynamic_axes=dynamic_axes)
        except RuntimeError as e:
            raise RuntimeError('There was an error while exporting the PyTorch model to ONNX: {}'.format(e))

        return onnx.load_model_from_string(f.getvalue())
