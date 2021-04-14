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
import os
import torch
import inspect
import numpy as np
from inspect import signature
from enum import IntEnum

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

def _create_forward_iobinding(io_binding, inputs, model, device, outputs):
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_ortvalue_input(
            value_info.name, _ortvalue_from_torch_tensor(inputs[idx]))

    for output_name in outputs:
        io_binding.bind_output(output_name, device.type,
                               device_id=_utils.get_device_index(device))

def _create_backward_iobinding(io_binding, inputs, model, device, input_names):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, name in enumerate(input_names):
        io_binding.bind_ortvalue_input(
            name, _ortvalue_from_torch_tensor(inputs[idx]))

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


def _load_torch_allocator_cpp_extension(verbosity):
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
                       functions=['cuda_caching_allocator_raw_alloc_address',
                                  'cuda_caching_allocator_raw_delete_address'],
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

                    # Use IO binding
                    # it is found for megatron training, the training input is non-contiguous.
                    contiguous_inputs = []
                    for idx, _input in enumerate(inputs):
                        if _input is None:
                            raise ValueError("find some of input is None")
                        elif not _input.is_contiguous():
                            _contiguous_input = _input.contiguous()
                        else:
                            _contiguous_input = _input
                        contiguous_inputs.append(_contiguous_input)

                    _create_forward_iobinding(self._training_foward_io_binding, contiguous_inputs, self._onnx_training, self._device, self._onnx_graphs_info.user_output_names)

                    # Run and return module outputs.
                    run_id = self._training_session.run_forward(self._training_foward_io_binding, self._run_options)
                    user_outputs = tuple(_ortvalue_to_torch_tensor(
                        forward_output) for forward_output in self._training_foward_io_binding.get_outputs())

                    ctx.run_id = run_id

                    # Disable materializing grads then None object will not be converted
                    # to a tensor filled with zeros prior to calling backward.
                    # Also save shape, device and type info to ctx for materializing
                    # tensor in backward if output grad is None.
                    ctx.set_materialize_grads(False)
                    ctx.output_info = [
                        (output.shape, output.device, output.dtype) for output in user_outputs]

                    # Assert that the outputs and model device match
                    _check_same_device(
                        self._device, "Output argument from forward", *user_outputs)

                    return user_outputs

                @staticmethod
                def backward(ctx, *grad_outputs):
                    '''Performs backward pass based on grad wrt module output
                    '''

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
                            shape, device, dtype = ctx.output_info[idx]
                            if idx in self._onnx_graphs_info.output_grad_indices_require_full_shape:
                                grad_output = torch.zeros(
                                    shape, device=device, dtype=dtype)
                            else:
                                grad_output = torch.tensor(
                                    0., device=device, dtype=dtype)
                        elif not grad_output.is_contiguous():
                            grad_output = grad_output.contiguous()
                        contiguous_grad_outputs.append(grad_output)

                    # Run and get results
                    run_id = ctx.run_id
                    _create_backward_iobinding(self._training_backward_io_binding, contiguous_grad_outputs, self._onnx_training, self._device, self._onnx_graphs_info.ort_break_op_output_names)
                    self._training_session.run_backward(self._training_backward_io_binding, self._run_options, np.int64(run_id))
                    backward_outputs = self._training_backward_io_binding.get_outputs()

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
                    self._training_backward_io_binding.clear_binding_outputs()
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
        self._training_foward_io_binding = None
        self._training_backward_io_binding = None
        self._run_options = None

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

        from torch.utils.cpp_extension import ROCM_HOME
        self.is_rocm_pytorch = (True if (
            (torch.version.hip is not None) and (ROCM_HOME is not None)) else False)

        # CPP extension to get torch CUDA allocator's alloc and free function addresses
        # Disable external allocator for ROCM EP since external allocator is not supported yet.
        self._use_external_cuda_allocator = (
            False if self.is_rocm_pytorch else True)
        if self._use_external_cuda_allocator:
            self._torch_cuda_allocator = _load_torch_allocator_cpp_extension(
                self._verbosity)
            self._torch_alloc = self._torch_cuda_allocator.cuda_caching_allocator_raw_alloc_address()
            self._torch_free = self._torch_cuda_allocator.cuda_caching_allocator_raw_delete_address()

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
        onnx_initializer_names = {
            p.name for p in self._onnx_inference.graph.initializer}
        initializer_names_to_train = [
            p for p in initializer_names_to_train if p in onnx_initializer_names]

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
            onnx.save(self._onnx_inference,
                      self._save_onnx_prefix + '_inference.onnx')

        self._initialize_module_gradient_graph_builder()

    def _create_training_session(self):
        providers = None
        provider_options = None
        if self._device.type == 'cuda':
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = (["ROCMExecutionProvider"] if self.is_rocm_pytorch else [
                         "CUDAExecutionProvider"])
            providers.append("CPUExecutionProvider")
            if self._use_external_cuda_allocator:
                provider_options = [{"device_id": str(self._device.index), "cuda_external_alloc": str(
                    self._torch_alloc), "cuda_external_free": str(self._torch_free)}, {}]
            else:
                provider_options = [{"device_id": str(self._device.index)}, {}]
        elif self._device.type == 'cpu':
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = False
        session_options.transfer_ownership_intermediate_output_tensors = False
        session_options.use_deterministic_compute = False
        # default to PRIORITY_BASED execution order
        session_options.execution_order = onnxruntime.ExecutionOrder.PRIORITY_BASED
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = int(self._verbosity)

        self._training_session = onnxruntime.training.TrainingAgent(self._onnx_training.SerializeToString(),
                                                                    session_options, providers, provider_options)

        # Use this global run_options for now
        self._run_options = C.RunOptions()

        # IO binding
        # TODO: we should try to reuse the output buffers as some of the output tensors are same sizes, expecially the backward graph outputs.
        self._training_foward_io_binding = self._training_session.io_binding()
        self._training_backward_io_binding = self._training_session.io_binding()

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
            onnx.save(self._onnx_training,
                      self._save_onnx_prefix + '_training.onnx')

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
        if len(non_none_inputs) != len(self._onnx_graphs_info.user_input_names):
            # temporary for fp16 training, ignore the attention_mask input, which is not used in fp16 training
            # todo: should we handle this in exporter?
            print("fp16, remove attention_mask input because it it not used")
            del non_none_inputs[2]

        result = []
        for input_idx, name in enumerate(self._onnx_graphs_info.user_input_names):
            inp = None
            if input_idx < len(non_none_inputs):
                inp = non_none_inputs[input_idx]
            elif name in kwargs and kwargs[name] is not None:
                inp = kwargs[name]
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
                                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                                  custom_opsets={"prim": 1},
                                  verbose=self._verbosity < Verbosity.WARNING)
        except RuntimeError as e:
            raise RuntimeError(
                'There was an error while exporting the PyTorch model to ONNX: {}'.format(e))

        my_model = onnx.load_model_from_string(f.getvalue())
        my_model.opset_import[1].domain = 'com.microsoft'
        index = 0
        for node in my_model.graph.node:
            if node.domain == 'prim':
                node.domain = 'com.microsoft'
                output_names = list(node.output)
                del node.output[:]
                node.output.append(output_names[0] + '_ctx')
                node.output.extend(output_names)
            if not node.name:
                # give a name for debugging
                node.name = node.op_type + "_id_" + str(index)
                index += 1

        onnx.save(my_model, 'my_model_new.onnx')
        initializer_names = [i.name for i in my_model.graph.initializer]
        exported_model_input_count = [i for i in my_model.graph.input if i.name not in initializer_names]
        if len(sample_inputs_copy) != len(exported_model_input_count):
            print("WARINING: exported model has inputs {}, while training inputs are {}".format(exported_model_input_count, sample_inputs_copy))
        if "FP16_TRAINING" in os.environ:
            # for fp16 training, workaround the naming mismatch between exported model parameters and pytorch named parameters.
            for p in my_model.graph.initializer:
                p.name = p.name.replace("_base_module.language_model", "_base_module.module.language_model")
            for p in my_model.graph.input:
                p.name = p.name.replace("_base_module.language_model", "_base_module.module.language_model")
            for node in my_model.graph.node:
                for i, _ in enumerate(node.input):
                    node.input[i] = node.input[i].replace("_base_module.language_model", "_base_module.module.language_model")
        onnx.save(my_model, 'my_model_rectified.onnx')
        return my_model
