# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Support for PyTorch C++ extensions within ORTModule

IMPORTANT: All extensions must explicitly use TORCH_CPP_BUILD_DIR as `build_directory`
           to allow ORTModule to monitor TORCH_CPP_BUILD_DIR/lock and warn the user
           when abnormal initialization occurs

TODO: Implement mechanism to register extensions and prevent issues with incorrect/missing flags
      for each :meth:`torch.utils.cpp_extension.load_inline` call
"""

import threading
from functools import wraps
from torch.utils.cpp_extension import load_inline

from onnxruntime.capi import _pybind_state as C
from onnxruntime.training.ortmodule import TORCH_CPP_BUILD_DIR


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

    return load_inline(name='torch_allocator',
                       cpp_sources=[torch_gpu_allocator_addresses_cpp_source],
                       extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                       functions=['gpu_caching_allocator_raw_alloc_address',
                                  'gpu_caching_allocator_raw_delete_address'],
                       verbose=verbosity,
                       with_cuda=True,
                       build_directory=TORCH_CPP_BUILD_DIR)

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
#include <tuple>
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
void SetIValueArguments(const std::vector<std::tuple<size_t, T>>& raw_arguments,
                        std::vector<c10::IValue>& ivalue_arguments) {
  for (size_t i = 0; i < raw_arguments.size(); i++) {
    size_t index = std::get<0>(raw_arguments[i]);
    TORCH_INTERNAL_ASSERT(index < ivalue_arguments.size());
    ivalue_arguments[index] = c10::IValue(std::get<1>(raw_arguments[i]));
  }
}

// TODO: Add more argument types, such as list type.
std::vector<DLManagedTensor*> ExecuteATenOperator(
    const char* op_name, const std::vector<std::tuple<size_t, DLManagedTensor*>>& tensor_arguments,
    const std::vector<std::tuple<size_t, int64_t>>& int_arguments,
    const std::vector<std::tuple<size_t, float>>& float_arguments,
    const std::vector<std::tuple<size_t, bool>>& bool_arguments) {
  std::string op_name_str(op_name);
  std::shared_ptr<torch::jit::Operator> op = ATenOperatorCache::Instance().GetOperator(op_name_str);

  // TODO: need to handle optional argument and arguments with default values.
  std::vector<c10::IValue> arguments;
  arguments.resize(op->schema().arguments().size());
  for (size_t i = 0; i < tensor_arguments.size(); i++) {
    size_t index = std::get<0>(tensor_arguments[i]);
    at::Tensor tensor = at::fromDLPack(std::get<1>(tensor_arguments[i]));
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

    aten_op_executor_cpp_extension = load_inline(name='aten_op_executor', cpp_sources=[aten_op_executor_cpp_source],
                                                 extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                                                 functions=['execute_aten_operator_address'],
                                                 verbose=verbosity, with_cuda=True,
                                                 build_directory=TORCH_CPP_BUILD_DIR)

    C.register_aten_op_executor(str(aten_op_executor_cpp_extension.execute_aten_operator_address()))

def _load_aten_op_executor_cpp_extension_if_needed(onnx_model, verbosity, is_rocm_pytorch):
    for node in onnx_model.graph.node:
        if node.op_type == 'ATenOp' and node.domain == 'com.microsoft':
            _load_aten_op_executor_cpp_extension(verbosity, is_rocm_pytorch)
            break
