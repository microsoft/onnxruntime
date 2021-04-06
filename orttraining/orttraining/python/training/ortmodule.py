# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils
from . import _ortmodule_output_transformation as _ortmodule_io
from onnxruntime.training import register_custom_ops_pytorch_exporter
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

import functools
import io
import logging
import onnx
import onnxruntime
import torch
import inspect
from inspect import signature
from enum import IntEnum
from typing import Iterator, Optional, Tuple

from torch.utils.dlpack import from_dlpack, to_dlpack
from torch.utils.cpp_extension import load_inline

# Needed to override PyTorch methods
from typing import TypeVar
T = TypeVar('T', bound='Module')


ONNX_OPSET_VERSION = 12


def _ortvalue_to_torch_tensor(ortvalue):
    # PyTorch's to_dlpack() uses same config for both torch.bool and torch.uint8,
    # and convert the config to torch.uint8 tensor duing from_dlpack().
    # So we need to convert the torch tensor to torch.bool type if OrtValue is bool tensor.
    torch_tensor = from_dlpack(ortvalue._ortvalue.to_dlpack())
    return torch_tensor.to(torch.bool) if ortvalue.data_type() == 'tensor(bool)' else torch_tensor


def _ortvalue_from_torch_tensor(torch_tensor):
    return OrtValue(C.OrtValue.from_dlpack(to_dlpack(torch_tensor), torch_tensor.dtype == torch.bool))


class Verbosity(IntEnum):
    VERBOSE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4

def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_ortvalue_input(
            value_info.name, _ortvalue_from_torch_tensor(inputs[idx]))

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type,
                               device_id=_utils.get_device_index(device))


def _check_same_device(device, argument_str, *args):
    '''Check that all tensor arguments in *args reside on the same device as the input device'''

    for arg in args:
        if arg is not None and isinstance(arg, torch.Tensor):
            arg_device = torch.device(arg.device)
            if arg_device != device:
                raise RuntimeError(
                    f"{argument_str} found on device {arg_device}, but expected it to be on module device {device}.")

def _load_torch_gpu_allocator_cpp_extension(verbosity, is_rocm_pytorch):
    gpu_identifier = "hip" if is_rocm_pytorch else "cuda"
    gpu_allocator_header = "HIPCachingAllocator" if is_rocm_pytorch else "CUDACachingAllocator"
    torch_gpu_allocator_addresses_cpp_source = f"#include <torch/extension.h>\n" \
    f"#include <c10/{gpu_identifier}/{gpu_allocator_header}.h>\n" \
    f"size_t gpu_caching_allocator_raw_alloc_address() {{\n" \
    f"    return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_alloc);\n" \
    f"}}\n" \
    f"size_t gpu_caching_allocator_raw_delete_address() {{\n" \
    f"    return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_delete);\n" \
    f"}}\n"

    return load_inline(name='inline_extension', cpp_sources=[torch_gpu_allocator_addresses_cpp_source],
                       extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                       functions=['gpu_caching_allocator_raw_alloc_address',
                                  'gpu_caching_allocator_raw_delete_address'],
                       verbose=verbosity < Verbosity.WARNING, with_cuda=True)


class ORTModule(torch.nn.Module):

    def __init__(self, module):
        assert isinstance(
            module, torch.nn.Module), "'module' must be a torch.nn.Module"

        # Create forward dynamically, so each ORTModule instance will have its own copy.
        # This is needed to be able to copy the forward signatures from the original PyTorch models
        # and possibly have different signatures for different instances.
        def _forward(self, *inputs, **kwargs):
            '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

            ONNX model is exported the first time this method is executed.
            Next, we build a full training graph with module_gradient_graph_builder.
            Finally, we instantiate the ONNX Runtime InferenceSession.
            '''
            # TODO: using pytorch for evaluation for now. We will use ORT for evaluation later.
            # TODO: If the model is being executed with the gradient disabled (inside torch.no_grad() context for example),
            # leverage pytorch model for now.
            if not self._is_training():
                return self._original_module(*inputs, **kwargs)

            # Exporting module to ONNX for the first time
            if not self._onnx_training:
                device_from_module = _utils.get_device_from_module(
                    self._original_module)
                if not self._device or self._device != device_from_module:
                    self._device = device_from_module
                    if not self._device:
                        raise RuntimeError(
                            'A device must be specified in the model or data!')
                self._get_inference_graph_and_init_gradient_graph_builder(
                    *inputs, **kwargs)

            # Flag to indicate whether the gradient_graph needs to be built
            build_gradient_graph = self._current_input_shape is None
            _, _, input_names_require_grad, new_input_shape = \
                _ortmodule_io.parse_inputs_for_onnx_export(
                    self._original_module_parameters, self._onnx_inference, *inputs, **kwargs)
            initializer_names_to_train_set_user_model = {name for name, param in
                self._flattened_output_module.named_parameters() if param.requires_grad}
            initializer_names_to_train_set_onnx_graph = set(self._onnx_graphs_info.initializer_names_to_train) \
                if self._onnx_graphs_info else None
            # If inputs requiring gradient change from forward to the next, the module_gradient_graph_builder
            # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
            if input_names_require_grad != self._input_names_require_grad or \
                initializer_names_to_train_set_user_model != initializer_names_to_train_set_onnx_graph:
                self._input_names_require_grad = input_names_require_grad
                self._initialize_module_gradient_graph_builder()
                # Trigger the rebuilding of the gradient graph
                build_gradient_graph = True

            if build_gradient_graph:
                self._current_input_shape = new_input_shape
                self._build_training_graph()
                self._create_training_session()

            module_device = _utils.get_device_from_module(
                self._original_module)
            if self._device != module_device:
                self._device = module_device
                self._create_training_session()

            class _ORTModuleFunction(torch.autograd.Function):
                '''Use a custom torch.autograd.Function to associate self.backward_graph as the
                gradient implementation for self.forward_graph.'''

                @staticmethod
                def forward(ctx, *inputs, **kwargs):
                    '''Performs forward pass based on user input and PyTorch initializer

                    Autograd Function's apply() doesn't support keyword arguments,
                    so `*inputs` has all the arguments - keyword arguments converted
                    to positional by the caller.

                    Module outputs are returned to the user
                    '''

                    # Assert that the input and model device match
                    _check_same_device(
                        self._device, "Input argument to forward", *inputs)

                    # TODO: Try to reuse the output buffers as some of the output tensors are same sizes,
                    #   especially the backward graph outputs.
                    training_io_binding = self._training_session.io_binding()
                    run_options = C.RunOptions()
                    
                    # Use IO binding
                    _create_iobinding(training_io_binding, inputs, self._onnx_training, self._device)

                    # Run and return module outputs.
                    forward_outputs, run_id = self._training_session.run_forward(training_io_binding, run_options)
                    user_outputs = tuple(_ortvalue_to_torch_tensor(
                        forward_output) for forward_output in forward_outputs)
                    # Disable materializing grads then None object will not be converted to a tensor filled with zeros prior to calling backward.
                    # Also save shape, device and type info to ctx for materializing tensor in backward if output grad is None.
                    ctx.set_materialize_grads(False)
                    output_info = [(output.shape, output.device, output.dtype) for output in user_outputs]
                    ctx.run_info = onnxruntime.training.RunStateInfo(run_id, run_options, training_io_binding, output_info)

                    # Assert that the outputs and model device match
                    _check_same_device(
                        self._device, "Output argument from forward", *user_outputs)

                    return user_outputs

                @staticmethod
                def backward(ctx, *grad_outputs):
                    '''Performs backward pass based on grad wrt module output
                    '''
                    assert ctx.run_info is not None, 'forward() or __call__() methods must be called before backward()'

                    # Assert that the grad_outputs and model device match
                    _check_same_device(
                        self._device, "Input argument to backward", *grad_outputs)

                    # Use IO binding
                    # Push user output grads to ONNX backend.
                    contiguous_grad_outputs = []
                    for idx, grad_output in enumerate(grad_outputs):
                        if idx in self._onnx_graphs_info.output_grad_indices_non_differentiable:
                            assert grad_output is None, "ORT found the {}-th module output '{}' is non-differentiable according to the onnx graph. " \
                                                        "However, the gradient value is still provided by torch's autograd engine." \
                                                        .format(idx, self._onnx_graphs_info.user_output_names[idx]) 
                            continue
                        
                        if grad_output is None:
                            shape, device, dtype = ctx.run_info.output_info[idx]
                            if idx in self._onnx_graphs_info.output_grad_indices_require_full_shape:
                                grad_output = torch.zeros(
                                    shape, device=device, dtype=dtype)
                            else:
                                grad_output = torch.tensor(
                                    0., device=device, dtype=dtype)
                        elif not grad_output.is_contiguous():
                            grad_output = grad_output.contiguous()
                        contiguous_grad_outputs.append(grad_output)
                    backward_grad_output_ortvalue = [_ortvalue_from_torch_tensor(
                        grad_output) for grad_output in contiguous_grad_outputs]

                    # Run and get results
                    run_id = ctx.run_info.run_id
                    training_io_binding = ctx.run_info.io_binding
                    self._training_session.run_backward(backward_grad_output_ortvalue, run_id)
                    backward_outputs = training_io_binding.get_outputs()

                    # Return input and initializer gradients
                    num_user_input_grads = len(self._input_names_require_grad)

                    results = []
                    for input_name in self._onnx_graphs_info.user_input_names:
                        try:
                            # Append to the results the backward output for each input that required grad
                            results.append(_ortvalue_to_torch_tensor(
                                backward_outputs[self._input_names_require_grad.index(input_name)]))
                        except ValueError:
                            # input_name is not found in the self._input_names_require_grad list
                            # Append None to results for each input that did not require grad
                            results.append(None)

                    # Append gradients of initializer to results
                    # Go over each initializer, check if it required grad and append to results accordingly
                    initializer_names_to_train_set = set(self._onnx_graphs_info.initializer_names_to_train) \
                        if self._onnx_graphs_info else None
                    initializer_index = num_user_input_grads
                    for initializer_name in self._onnx_graphs_info.initializer_names:
                        if initializer_name in initializer_names_to_train_set:
                            results.append(_ortvalue_to_torch_tensor(backward_outputs[initializer_index]))
                            initializer_index += 1
                        else:
                            results.append(None)

                    # The OrtValue has a shared_ptr to the data.
                    # At this point there are two shared_ptrs to the data, one through the
                    # OrtValue in the output iobinding, and the other through the copy in OrtDLManagedTensor.
                    # The following call clears the iobinding output, reducing the use_count to 1, so that once torch finishes computation
                    # on the DLpack tensors, the memory can be freed.
                    training_io_binding.clear_binding_outputs()
                    return tuple(results)

            return _ortmodule_io.populate_user_output_from_schema_and_outputs(
                self._original_module_output_schema,
                self._onnx_graphs_info.user_output_names,
                _ORTModuleFunction.apply(*self._convert_training_graph_input_to_list(*inputs, **kwargs)))

        # Bind the forward method.
        self.forward = _forward.__get__(self)
        # Copy the forward signature from the PyTorch module.
        functools.update_wrapper(
            self.forward.__func__, module.forward.__func__)

        super(ORTModule, self).__init__()

        # Verbosity for logging
        self._verbosity = Verbosity.WARNING

        # Support contrib OPs
        register_custom_ops_pytorch_exporter.register_custom_op()

        # TODO: Single device support for now
        self._device = _utils.get_device_from_module(module)

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        # Get the module that flattens the output from the original module into a tuple
        self._flattened_output_module = \
            _ortmodule_io.get_flattened_output_module(
                self._original_module)
        self._original_module_parameters = signature(
            self._original_module.forward).parameters.values()

        # TODO: remove after PyTorch ONNX exporter supports VAR_KEYWORD parameters.
        for input_parameter in self._original_module_parameters:
            if input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
                raise NotImplementedError(
                    "The model's forward method has **kwargs parameter which is currently not supported.")

        self._onnx_inference = None

        # Related to training graph shape inference
        self._current_input_shape = None
        # default execution order is priority-based for both dynamic/static shape input for now
        # if we observe benefit of static shape, we can expose this flag to user
        self._use_static_shape = False
        self._module_gradient_graph_builder = None
        self._input_names_require_grad = None
        self._original_module_output_schema = None
        self._onnx_graphs_info = None

        # Training model
        self._onnx_training = None
        self._training_session = None

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

        from torch.utils.cpp_extension import ROCM_HOME
        self.is_rocm_pytorch = (True if (
            (torch.version.hip is not None) and (ROCM_HOME is not None)) else False)

        self._use_external_gpu_allocator = True
        if self._use_external_gpu_allocator:
            # CPP extension to get torch GPU allocator's alloc and free function addresses
            self._torch_gpu_allocator = _load_torch_gpu_allocator_cpp_extension(self._verbosity, self.is_rocm_pytorch)
            self._torch_alloc = self._torch_gpu_allocator.gpu_caching_allocator_raw_alloc_address()
            self._torch_free = self._torch_gpu_allocator.gpu_caching_allocator_raw_delete_address()

    def _is_training(self):
        return self._flattened_output_module.training and torch.is_grad_enabled()

    def _initialize_module_gradient_graph_builder(self):
        # TODO: PyTorch exporter bug: changes the initializer order in ONNX model
        initializer_names = [name
                             for name, _ in self._flattened_output_module.named_parameters()]
        initializer_names_to_train = []
        if self._is_training():
            initializer_names_to_train = [name
                for name, param in self._flattened_output_module.named_parameters() if param.requires_grad]

        # Build full training graph
        grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()
        grad_builder_config.initializer_names = initializer_names
        grad_builder_config.initializer_names_to_train = initializer_names_to_train
        grad_builder_config.input_names_require_grad = self._input_names_require_grad
        self._module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
        self._module_gradient_graph_builder.initialize(
            self._onnx_inference.SerializeToString(), grad_builder_config)

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
            providers = (["ROCMExecutionProvider"] if self.is_rocm_pytorch else [
                         "CUDAExecutionProvider"])
            providers.append("CPUExecutionProvider")
            if self._use_external_gpu_allocator:
                provider_options = [{"device_id": str(self._device.index), "gpu_external_alloc": str(
                    self._torch_alloc), "gpu_external_free": str(self._torch_free)}, {}]
            else:
                provider_options = [{"device_id": str(self._device.index)}, {}]
        elif self._device.type == 'cpu':
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.use_deterministic_compute = False
        # default to PRIORITY_BASED execution order
        session_options.execution_order = onnxruntime.ExecutionOrder.PRIORITY_BASED
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = int(self._verbosity)
        # enable dumping optimized training graph
        if self._save_onnx:
            session_options.optimized_model_filepath = self._save_onnx_prefix + '_training_optimized.onnx'

        self._training_session = onnxruntime.training.TrainingAgent(self._onnx_training.SerializeToString(),
                                                                    session_options, providers, provider_options)

    def _build_training_graph(self, *inputs, **kwargs):
        if self._use_static_shape:
            self._module_gradient_graph_builder.build(
                self._current_input_shape)
        else:
            self._module_gradient_graph_builder.build()
        self._onnx_training = onnx.load_model_from_string(
            self._module_gradient_graph_builder.get_training_model())
        self._onnx_graphs_info = self._module_gradient_graph_builder.get_training_graph_info()

        if self._save_onnx:
            inference_optimized_model = onnx.load_model_from_string(
                self._module_gradient_graph_builder.get_inference_optimized_model())
            onnx.save(inference_optimized_model, self._save_onnx_prefix + '_inference_optimized.onnx')
            onnx.save(self._onnx_training, self._save_onnx_prefix + '_training.onnx')

    def eval(self: T) -> T:
        self._flattened_output_module.eval()

    def train(self: T, mode: bool = True) -> T:
        self._flattened_output_module.train(mode)

    def _convert_training_graph_input_to_list(self, *inputs, **kwargs):
        '''Creates forward `*inputs` list from user input and PyTorch initializers

        TODO: How IO binding model inputs and outputs affects initializer copies?

        ONNX Runtime forward requires an ordered list of:
            * User input: computed from forward InferenceSession
            * Initializers: computed from original PyTorch model parameters
        '''
        # User inputs
        non_none_inputs = [inp for inp in inputs if inp is not None]
        named_buffers_iter = iter(self._flattened_output_module.named_buffers())
        result = []
        for input_idx, name in enumerate(self._onnx_graphs_info.user_input_names):
            inp = None
            if input_idx < len(non_none_inputs):
                inp = non_none_inputs[input_idx]
            elif name in kwargs and kwargs[name] is not None:
                inp = kwargs[name]
            elif input_idx >= len(non_none_inputs):
                # Registered buffers are translated to user_input+initializer in ONNX
                # TODO: Check what happens when the number of inputs change form one call to the next
                buffer_name, inp = next(named_buffers_iter)
                assert buffer_name == name, f'Input name {name} expected, but {buffer_name} found!'

            if inp is not None:
                result.append(inp)
            else:
                # TODO: Re-export ONNX if any input from _onnx_graphs_info.user_input_names is None.
                raise RuntimeError(
                    f'Input is present in ONNX graph but not provided: {name}.')

        # Initializers
        for param in self._flattened_output_module.named_parameters():
            result.append(param[1])

        return result

    def _get_inference_graph(self, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        '''

        # Setup dynamic axes for onnx model
        input_names, dynamic_axes, self._input_names_require_grad, _ = \
            _ortmodule_io.parse_inputs_for_onnx_export(
                self._original_module_parameters, None, *inputs, **kwargs)
        output_names, output_dynamic_axes, self._original_module_output_schema = \
            _ortmodule_io.parse_outputs_for_onnx_export_and_extract_output_schema(
                self._original_module, inputs, kwargs)
        dynamic_axes.update(output_dynamic_axes)

        # Export torch.nn.Module to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        sample_inputs_copy, sample_kwargs_copy = \
            _ortmodule_io.deepcopy_model_input(
                *inputs, **kwargs)

        try:
            with torch.no_grad():
                torch.onnx.export(self._flattened_output_module,
                                  sample_inputs_copy + (sample_kwargs_copy, ),
                                  f,
                                  input_names=input_names,
                                  output_names=output_names,
                                  opset_version=ONNX_OPSET_VERSION,
                                  do_constant_folding=False,
                                  training=torch.onnx.TrainingMode.TRAINING,
                                  dynamic_axes=dynamic_axes,
                                  verbose=self._verbosity < Verbosity.WARNING,
                                  export_params=False,
                                  keep_initializers_as_inputs=True)
        except RuntimeError as e:
            raise RuntimeError(
                'There was an error while exporting the PyTorch model to ONNX: {}'.format(e))

        return onnx.load_model_from_string(f.getvalue())

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override original method to delegate execution to the base module"""

        # Override the state_dict() method so that the state dict key names
        # do not contain the _flattened_output_module._base_module prefix
        return self._flattened_output_module._base_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Override original method to delegate execution to the base module"""

        # Override the load_state_dict() method so that the loaded state dict
        # key names does not need to contain the _flattened_output_module._base_module prefix
        return self._flattened_output_module._base_module.load_state_dict(
            state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Override original method to delegate execution to the base module"""
        self._flattened_output_module._base_module.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Override original method to delegate execution to the base module"""
        self._flattened_output_module._base_module.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Override original method to delegate execution to the base module"""
        return self._flattened_output_module._base_module.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        """Override original method to delegate execution to the base module"""
        return self._flattened_output_module._base_module.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Override original method to delegate execution to the base module"""
        yield from self._flattened_output_module._base_module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override original method to delegate execution to the base module"""
        yield from self._flattened_output_module._base_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override original method to delegate execution to the base module"""
        yield from self._flattened_output_module._base_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override original method to delegate execution to the base module"""
        yield from self._flattened_output_module._base_module.named_buffers(prefix=prefix, recurse=recurse)