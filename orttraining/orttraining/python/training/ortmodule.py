import functools
import io
import logging
import onnx
import onnxruntime
import torch
from inspect import signature

from torch.utils.dlpack import from_dlpack
from torch.utils.cpp_extension import load_inline

# Needed to re-implement PyTorch's cpu,cuda,to methods
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict

from onnxruntime.capi import _pybind_state as C
from onnxruntime.training import register_custom_ops_pytorch_exporter
from . import _utils, _ortmodule_output_transformation


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

        # Create forward dynamically, so each ORTModule instance will have its own copy.
        # This is needed to be able to copy the forward signatures from the original PyTorch models
        # and possibly have different signatures for different instances.
        def _forward(self, *inputs, **kwargs):
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
                device_from_module = _utils.get_device_from_module(self._original_module)
                if not self._device or self._device != device_from_module:
                    self._device = device_from_module
                    if not self._device:
                        raise RuntimeError('A device must be specified in the model or data!')
                self._get_inference_graph_and_init_gradient_graph_builder(*inputs, **kwargs)

            _, _, input_names_require_grad, new_input_shape = \
                _ortmodule_output_transformation.parse_inputs_for_onnx_export(
                    self._original_module_input_names, self._onnx_inference, *inputs, **kwargs)
            # If inputs requiring gradient change from one call to forward to the next, the module_gradient_graph_builder
            # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
            if input_names_require_grad != self._input_names_require_grad:
                self._input_names_require_grad = input_names_require_grad
                self._initialize_module_gradient_graph_builder()

            if self._current_input_shape is None or self._current_input_shape != new_input_shape:
                self._current_input_shape = new_input_shape
                self._build_training_graph()
                self._create_training_session()

            module_device = _utils.get_device_from_module(self._original_module)
            if self._device != module_device:
                self._device = module_device
                self._create_training_session()


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
                    forward_outputs, run_id = self._training_session.run_forward(self._training_io_binding, self._run_options)
                    user_outputs = tuple(_ort_output_to_torch_tensor(forward_output) for forward_output in forward_outputs)
                    ctx.run_id = run_id

                    return user_outputs

                @staticmethod
                def backward(ctx, *grad_outputs):
                    '''Performs backward pass based on grad wrt module output
                    '''

                    # Use IO binding
                    # Push user output grads to ONNX backend.
                    backward_grad_output_ortvalue = []

                    # backward_output_grad_names_map only contains the subset of module outputs that need a gradient,
                    # we filter out the invalid entries in grad_outputs, accessing using the mapped index.

                    for _, i in self._onnx_graphs_info.backward_output_grad_names_map.items():
                        grad_output = grad_outputs[i]
                        if not grad_output.is_contiguous():
                            grad_output = grad_output.contiguous()
                        backward_grad_output_ortvalue.append(onnxruntime.OrtValue.ortvalue_from_data_ptr(list(grad_output.size()), _utils.dtype_torch_to_numpy(
                            grad_output.dtype), grad_output.device.type, _utils.get_device_index(grad_output.device), grad_output.data_ptr()))

                    # Run and get results
                    run_id = ctx.run_id
                    self._training_session.run_backward(backward_grad_output_ortvalue, run_id)
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
                    # The OrtValue has a shared_ptr to the data. At this point there are two shared_ptrs to the data, one through the 
                    # OrtValue in the output iobinding, and the other through the copy in OrtDLManagedTensor.
                    # The following call clears the iobinding output, reducing the use_count to 1, so that once torch finishes computation
                    # on the DLpack tensors, the memory can be freed.
                    self._training_io_binding.clear_binding_outputs()
                    return tuple(results)

            return _ortmodule_output_transformation.populate_user_output_from_schema_and_outputs(self._original_module_output_schema,
                self._onnx_graphs_info.user_output_names,
                _ORTModuleFunction.apply(*self._convert_training_graph_input_to_list(*inputs, **kwargs)))

        # Bind the forward method.
        self.forward = _forward.__get__(self)
        # Copy the forward signature from the PyTorch module.
        functools.update_wrapper(self.forward.__func__, module.forward.__func__)

        super(ORTModule, self).__init__()

        # Support contrib OPs
        register_custom_ops_pytorch_exporter.register_custom_op()

        # TODO: Single device support for now
        self._device = _utils.get_device_from_module(module)

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        # Get the module that flattens the output from the original module into a tuple
        self._flattened_output_module = \
            _ortmodule_output_transformation.get_flattened_output_module(self._original_module)
        sig = signature(self._original_module.forward)
        self._original_module_input_names = sig.parameters.keys()
        self._onnx_inference = None
        self._is_training = True

        # Related to training graph shape inference
        self._current_input_shape = None
        self._module_gradient_graph_builder = None
        self._input_names_require_grad = None
        self._original_module_output_schema = None

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
        # TODO: PyTorch exporter bug: changes the initializer order in ONNX model
        initializer_names = [p[0] for p in self._flattened_output_module.named_parameters()]
        onnx_initializer_names = [p.name for p in self._onnx_inference.graph.initializer]
        initializer_names = [p for p in initializer_names if p in onnx_initializer_names]

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

    def eval(self: T) -> T:
        self._is_training = False
        self._flattened_output_module.eval()

    def train(self: T, mode: bool = True) -> T:
        self._is_training = mode
        self._flattened_output_module.train(mode)

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
        for param in self._flattened_output_module.named_parameters():
            result.append(param[1])

        return result

    def _get_inference_graph(self, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        TODO: How to ingest **kwargs in proper order during export?
        '''

        # Setup dynamic axes for onnx model
        input_names, dynamic_axes, self._input_names_require_grad, _ = \
            _ortmodule_output_transformation.parse_inputs_for_onnx_export(
                self._original_module_input_names, None, *inputs, **kwargs)
        output_names, output_dynamic_axes, self._original_module_output_schema = \
            _ortmodule_output_transformation.parse_outputs_for_onnx_export_and_extract_output_schema(
                self._original_module, inputs, kwargs)
        dynamic_axes.update(output_dynamic_axes)

        # Export torch.nn.Module to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        sample_inputs_copy, sample_kwargs_copy = \
            _ortmodule_output_transformation.deepcopy_model_input(*inputs, **kwargs)

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
                                dynamic_axes=dynamic_axes)
        except RuntimeError as e:
            raise RuntimeError('There was an error while exporting the PyTorch model to ONNX: {}'.format(e))

        return onnx.load_model_from_string(f.getvalue())
