# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

import threading
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch.utils.cpp_extension import load_inline
from functools import wraps


def _ortvalue_to_torch_tensor(ortvalue):
    # PyTorch's to_dlpack() uses same config for both torch.bool and torch.uint8,
    # and convert the config to torch.uint8 tensor duing from_dlpack().
    # So we need to convert the torch tensor to torch.bool type if OrtValue is bool tensor.
    torch_tensor = from_dlpack(ortvalue.to_dlpack())
    return torch_tensor.to(torch.bool) if ortvalue.data_type() == 'tensor(bool)' else torch_tensor

def _ortvalue_from_torch_tensor(torch_tensor):
    return C.OrtValue.from_dlpack(to_dlpack(torch_tensor), torch_tensor.dtype == torch.bool)

def _load_torch_gpu_allocator_cpp_extension(verbosity, is_rocm_pytorch):
    gpu_identifier = "hip" if is_rocm_pytorch else "cuda"
    gpu_allocator_header = "HIPCachingAllocator" if is_rocm_pytorch else "CUDACachingAllocator"
    torch_gpu_allocator_addresses_cpp_source = f'''
        #include <torch/extension.h>
        #include <c10/{gpu_identifier}/{gpu_allocator_header}.h>

        size_t gpu_caching_allocator_raw_alloc_address() {{
            return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_alloc);
        }}

        size_t gpu_caching_allocator_raw_delete_address() {{
            return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_delete);
        }}
    '''

    return load_inline(name='inline_extension',
                       cpp_sources=[torch_gpu_allocator_addresses_cpp_source],
                       extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                       functions=['gpu_caching_allocator_raw_alloc_address',
                                  'gpu_caching_allocator_raw_delete_address'],
                       verbose=verbosity,
                       with_cuda=True)


def _check_same_device(device, argument_str, *args):
    '''Check that all tensor arguments in *args reside on the same device as the input device'''

    assert isinstance(device, torch.device), '`device` must be a valid `torch.device` object'
    for arg in args:
        if arg is not None and isinstance(arg, torch.Tensor):
            arg_device = torch.device(arg.device)
            if arg_device != device:
                raise RuntimeError(
                    f"{argument_str} found on device {arg_device}, but expected it to be on module device {device}.")


def get_device_index(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        device = torch.device(device)
    elif isinstance(device, int):
        return device
    return 0 if device.index is None else device.index


def get_device_str(device):
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
        raise RuntimeError('Unsupported device type')
    return device


def get_device_from_module(module):
    '''Returns the first device found in the `module`'s parameters or None'''
    device = None
    try:
        device = next(module.parameters()).device
        for param in module.parameters():
            if param.device != device:
                raise RuntimeError('ORTModule supports a single device per model for now')
    except StopIteration:
        # Model doesn't have a device set to any of the model parameters
        pass
    return device

def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_ortvalue_input(value_info.name, OrtValue(_ortvalue_from_torch_tensor(inputs[idx])))

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type, device_id=get_device_index(device))

def run_once_aten_op_executor(f):
    """
    Decorator to run a function only once.
    :param f: function to be run only once during execution time despite the number of calls
    :return: The original function with the params passed to it if it hasn't already been run before
    """
    @wraps(f)
    def aten_op_executor_wrapper(*args, **kwargs):
        if not aten_op_executor_wrapper.has_run:
            with aten_op_executor_wrapper.lock:
                if not aten_op_executor_wrapper.has_run:
                    aten_op_executor_wrapper.has_run = True
                    return f(*args, **kwargs)

    aten_op_executor_wrapper.lock = threading.Lock()
    aten_op_executor_wrapper.has_run = False
    return aten_op_executor_wrapper

@run_once_aten_op_executor
def _load_aten_op_executor_cpp_extension(verbosity, is_rocm_pytorch):
    aten_op_executor_cpp_source = """
#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <unordered_map>
#include <vector>

class ATenOperatorCache {
 public:
  static ATenOperatorCache& Instance() {
    static ATenOperatorCache instance;
    return instance;
  }

  std::shared_ptr<torch::jit::Operator> GetOperator(const std::string& op_name) {
    if (ops_.find(op_name) == ops_.end()) {
      auto& ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString(op_name));
      TORCH_INTERNAL_ASSERT(ops.size() == 1);
      ops_[op_name] = ops.front();
    }

    return ops_.at(op_name);
  }

 private:
  ATenOperatorCache() = default;
  std::unordered_map<std::string, std::shared_ptr<torch::jit::Operator>> ops_;
};

// Some arguments of backward operator are not from forward operator's input or output,
// but need some processing. Since we cannot build such processing to ONNX graph for now,
// we are putting such processing code here if needed.
// Take embedding_backward as example:
//   weight: embedding_backward(grad, indices, weight.size(0), padding_idx, scale_grad_by_freq, sparse)
// the 3rd argument (index 2) is weight.size(0), we add this processing here.
using TensorTransformFunc = std::function<c10::IValue(const at::Tensor&)>;
static const TensorTransformFunc embedding_num_weights = [](const at::Tensor& tensor) {
  return c10::IValue(tensor.size(0));
};

static const std::unordered_map<std::string, std::unordered_map<size_t, TensorTransformFunc>> TENSOR_TRANSFORM_FUNCS = {
    {"aten::embedding_backward", {{2, embedding_num_weights}}},
};

template <typename T>
void SetIValueArguments(const std::vector<std::pair<size_t, T>>& raw_arguments,
                        std::vector<c10::IValue>& ivalue_arguments) {
  for (const auto& raw_argument : raw_arguments) {
    size_t index = raw_argument.first;
    TORCH_INTERNAL_ASSERT(index < ivalue_arguments.size());
    ivalue_arguments[index] = c10::IValue(raw_argument.second);
  }
}

// TODO: Add more argument types, such as list type.
std::vector<DLManagedTensor*> ExecuteATenOperator(
    const char* op_name, const std::vector<std::pair<size_t, DLManagedTensor*>>& tensor_arguments,
    const std::vector<std::pair<size_t, int64_t>>& int_arguments,
    const std::vector<std::pair<size_t, float>>& float_arguments,
    const std::vector<std::pair<size_t, bool>>& bool_arguments) {
  std::string op_name_str(op_name);
  std::shared_ptr<torch::jit::Operator> op = ATenOperatorCache::Instance().GetOperator(op_name_str);

  // TODO: need to handle optional argument and arguments with default values.
  std::vector<c10::IValue> arguments;
  arguments.resize(op->schema().arguments().size());
  for (const auto& tensor_argument : tensor_arguments) {
    size_t index = tensor_argument.first;
    at::Tensor tensor = at::fromDLPack(tensor_argument.second);
    bool has_transform_func = false;
    if (TENSOR_TRANSFORM_FUNCS.find(op_name_str) != TENSOR_TRANSFORM_FUNCS.end()) {
      const auto& transform_funcs = TENSOR_TRANSFORM_FUNCS.at(op_name_str);
      if (transform_funcs.find(index) != transform_funcs.end()) {
        arguments[index] = transform_funcs.at(index)(tensor);
        has_transform_func = true;
      }
    }

    if (!has_transform_func) {
      arguments[index] = c10::IValue(tensor);
    }
  }

  SetIValueArguments<int64_t>(int_arguments, arguments);
  SetIValueArguments<float>(float_arguments, arguments);
  SetIValueArguments<bool>(bool_arguments, arguments);

  torch::jit::Stack stack;
  for (size_t i = 0; i < arguments.size(); i++) {
    torch::jit::push(stack, arguments[i]);
  }

  op->getOperation()(&stack);
  // TODO: need to handle multiple-tensor outputs.
  at::Tensor output;
  torch::jit::pop(stack, output);
  std::vector<DLManagedTensor*> result;
  result.emplace_back(at::toDLPack(output));
  return result;
}

size_t execute_aten_operator_address() { return reinterpret_cast<size_t>(&ExecuteATenOperator); }
    """

    aten_op_executor_cpp_extension = load_inline(name='inline_extension_aten_op_executor', cpp_sources=[aten_op_executor_cpp_source],
                                                 extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                                                 functions=['execute_aten_operator_address'],
                                                 verbose=verbosity, with_cuda=True)

    C.register_aten_op_executor(str(aten_op_executor_cpp_extension.execute_aten_operator_address()))

def _load_aten_op_executor_cpp_extension_if_needed(onnx_model, verbosity, is_rocm_pytorch):
    for node in onnx_model.graph.node:
        if node.op_type == 'ATenOp' and node.domain == 'com.microsoft':
            _load_aten_op_executor_cpp_extension(verbosity, is_rocm_pytorch)
            break
