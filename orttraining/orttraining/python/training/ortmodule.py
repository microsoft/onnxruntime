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

# Needed to re-implement PyTorch's cpu,cuda,to methods
from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict

from onnxruntime.capi import _pybind_state as C
from . import _utils


ONNX_OPSET_VERSION = 12
__TEMP_ENABLE_METHOD_TIMING__ = True

# Needed to re-implement PyTorch's cpu,cuda,to methods
T = TypeVar('T', bound='Module')

def _create_iobinding(io_binding, inputs, model, output_buffers, device):
    '''Creates IO binding for a `model` inputs and output'''
    def _get_device_index(device):
        if type(device) == str:
            # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
            device = torch.device(device)
        return 0 if device.index is None else device.index

    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_input(value_info.name, inputs[idx].device.type,
                              _get_device_index(inputs[idx].device),
                              _utils.dtype_torch_to_numpy(inputs[idx].dtype),
                              list(inputs[idx].size()),
                              inputs[idx].data_ptr())

    for value_info in model.graph.output:
        name = value_info.name
        output_tensor = output_buffers[name]
        io_binding.bind_output(name, output_tensor.device.type,
                               _get_device_index(device),
                               _utils.dtype_torch_to_numpy(output_tensor.dtype),
                               list(output_tensor.size()),
                               output_tensor.data_ptr())

def _onnx_value_info_to_buffer_tensor(value_info, device):
    '''Create a torch zeroed tensor with the same shape and type of `value_info`'''

    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    dtype = _utils.dtype_onnx_to_torch(value_info.type.tensor_type.elem_type)
    return torch.zeros(shape, device=device, dtype=dtype)

class ORTModule(torch.nn.Module):

    def __init__(self, module, dynamic_axes=None):
        assert isinstance(module, torch.nn.Module), "'module' mst be a torch.nn.Module"
        super(ORTModule, self).__init__()

        self._export_again = False
        # TODO: This is incorrect when different layers may be in different devices
        self._device = next(module.parameters()).device

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        self._dynamic_axes = dynamic_axes
        self._onnx_training = None

        self._curr_inputs_size = None
        self._module_gradient_graph_builder = None

        # Forward pass
        self._onnx_forward = None
        self._forward_session = None
        self._forward_io_binding = None
        self._forward_output_buffers = {}

        # Backward pass
        self._onnx_backward = None
        self._backward_session = None
        self._backward_io_binding = None
        self._backward_output_buffers = {}

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # Debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''

    def cpu(self: T) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''
        if self._device != 'cpu':
            self._require_export = True
            self._device = 'cpu'
        return super(ORTModule, self).cpu()

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''
        if device:
            device = str(device)
        else:
            device = 'cuda'
        if self._device != str(device):
            self._require_export = True
            self._device = device
        return super(ORTModule, self).cuda(device)

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ...,
           dtype: Optional[Union[dtype, str]] = ...,
           non_blocking: bool = ...) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''
        if device:
            device = str(device)
        else:
            device = None
        if self._device != str(device) and device is not None:
            self._require_export = True
            self._device = device
        return super(ORTModule, self).to(device, dtype, non_blocking)

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''
        # TODO: Should we do anything?
        self._require_export = False
        return super(ORTModule, self).to(dtype, non_blocking)

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        '''Thin layer to capture device for ORTModule IO bindings'''
        # TODO: Long shot by sending model to tensor's device
        device = None
        if tensor:
            device = str(tensor.device)
        if self._device != str(device) and device is not None:
            self._require_export = True
            self._device = device
        return super(ORTModule, self).to(tensor, non_blocking)

    def to(self, *args, **kwargs):
        '''Thin layer to capture device for ORTModule IO bindings'''
        # TODO: Should we do anything?
        self._require_export = False
        return super(ORTModule, self).to(args, kwargs)

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, a full training graph is splitted in forward and backward graph which are used
        to instantiate ONNX Runtime InferenceSession`s
        '''
        if not self._onnx_forward or self._require_export:
            self._require_export = False

            self._onnx_training = ORTModule._get_forward_graph(self._original_module, self._dynamic_axes, *inputs, **kwargs)
            grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()

            # TODO: PyTorch exporter bug: changes the initializer order
            initializer_names = [p[0] for p in self._original_module.named_parameters()]
            grad_builder_config.initializer_names_to_train = initializer_names
            grad_builder_config.input_names_require_grad = []
            self._module_gradient_graph_builder = C.ModuleGradientGraphBuilder()
            self._module_gradient_graph_builder.initialize(self._onnx_training.SerializeToString(), grad_builder_config)

            if self._save_onnx:
                onnx.save(self._onnx_training, self._save_onnx_prefix + '_full_training.onnx')

        inputs_size = [list(input.size()) for input in inputs if input is not None]
        if self._curr_inputs_size is None or self._curr_inputs_size != inputs_size:
            self._curr_inputs_size = inputs_size
            self._module_gradient_graph_builder.build_and_split(self._curr_inputs_size)
            self._onnx_forward = onnx.load_model_from_string(self._module_gradient_graph_builder.get_forward_model())
            self._onnx_backward = onnx.load_model_from_string(self._module_gradient_graph_builder.get_backward_model())
            self._onnx_graphs_info = self._module_gradient_graph_builder.get_split_graphs_info()

            if self._save_onnx:
                onnx.save(self._onnx_forward, self._save_onnx_prefix + '_forward.onnx')
                onnx.save(self._onnx_backward, self._save_onnx_prefix + '_backward.onnx')

            self._forward_session = onnxruntime.InferenceSession(self._onnx_forward.SerializeToString())
            self._backward_session = onnxruntime.InferenceSession(self._onnx_backward.SerializeToString())

            # IO binding
            # TODO: we should try to reuse the output buffers as some of the output tensors are same sizes, expecially the backward graph outputs.
            self._forward_io_binding = self._forward_session.io_binding()
            self._forward_output_buffers = {}
            for output in self._onnx_forward.graph.output:
                self._forward_output_buffers[output.name] = _onnx_value_info_to_buffer_tensor(output, str(self._device))
            self._backward_io_binding = self._backward_session.io_binding()
            self._backward_output_buffers = {}
            for output in self._onnx_backward.graph.output:
                self._backward_output_buffers[output.name] = _onnx_value_info_to_buffer_tensor(output, str(self._device))

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
                                  self._forward_output_buffers,
                                  str(self._device))

                # Run
                self._forward_session.run_with_iobinding(self._forward_io_binding)

                # Stash tensors needed by backward
                forward_input_dict = self._convert_forward_input_list_to_dict(*inputs)
                ctx_inputs = tuple(forward_input_dict[name] \
                    for name in self._onnx_graphs_info.backward_user_input_names)
                ctx_initializers = tuple(forward_input_dict[name] \
                    for name in self._onnx_graphs_info.backward_intializer_names_as_input)
                ctx_intermediates = tuple(self._forward_output_buffers[name] \
                    for name in self._onnx_graphs_info.intermediate_tensor_names)
                ctx.save_for_backward(*[*ctx_inputs, *ctx_initializers, *ctx_intermediates])

                # Return model output
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

                # Use IO binding
                grad_output_dict = dict(zip(self._onnx_graphs_info.user_output_grad_names, grad_output))
                backward_grad_output = tuple(grad_output_dict[name] for name in self._onnx_graphs_info.backward_output_grad_names)
                _create_iobinding(self._backward_io_binding, [*ctx.saved_tensors, *backward_grad_output],
                                   self._onnx_backward,
                                   self._backward_output_buffers,
                                   str(self._device))

                # Run
                self._backward_session.run_with_iobinding(self._backward_io_binding)

                # Return input and initializer gradients
                results = [torch.tensor([1])] * len(self._onnx_graphs_info.user_input_names)
                results += [self._backward_output_buffers[name] \
                    for name in self._onnx_graphs_info.initializer_grad_names_to_train]
                return tuple(results)

        proc_inputs = [data for data in inputs if data is not None]
        return _ORTModuleFunction.apply(*self._convert_forward_input_to_list(*proc_inputs, **kwargs))

    @_utils.timeit(enabled=__TEMP_ENABLE_METHOD_TIMING__)
    def _convert_forward_input_to_list(self, *inputs, **kwargs):
        '''Creates forward `*inputs` list from user input and PyTorch initializers

        TODO: **kwargs is not supported
        TODO: How IO binding model inputs and outputs affects initializer copies?

        ONNX Runtime forward requires an order list of:
            * User input: computed from ONNX forward graph, excluding initializers as input
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


    @staticmethod
    def _get_forward_graph(module, dynamic_axes, *inputs, **kwargs):
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
                          training=torch.onnx.TrainingMode.TRAINING,
                          dynamic_axes=dynamic_axes)

        return onnx.load_model_from_string(f.getvalue())
