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
#include <vector>

struct ATenOperator {
  std::shared_ptr<torch::jit::Operator> op;
  size_t argument_size;
  std::vector<bool> is_optional_arguments;
  size_t return_size;
};

class ATenOperatorCache {
 public:
  static ATenOperatorCache& Instance() {
    static ATenOperatorCache instance;
    return instance;
  }

  const ATenOperator& GetOperator(const std::string& op_name) {
    if (ops_.find(op_name) == ops_.end()) {
      auto& ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString(op_name));
      bool found = false;
      ATenOperator aten_op;
      for (auto op : ops) {
        // Some op name can get multiple ops with different overload names,
        // we are using the one without overload name.
        if (op->schema().overload_name() == "") {
          aten_op.op = op;
          found = true;
          break;
        }
      }

      TORCH_INTERNAL_ASSERT(found);
      const auto& schema = aten_op.op->schema();
      aten_op.argument_size = schema.arguments().size();
      for (const auto& argument : schema.arguments()) {
        aten_op.is_optional_arguments.emplace_back(argument.type()->kind() == c10::TypeKind::OptionalType);
      }

      aten_op.return_size = schema.returns().size();
      for (const auto& ret : schema.returns()) {
        TORCH_INTERNAL_ASSERT(ret.type()->kind() == c10::TypeKind::TensorType);
      }

      ops_[op_name] = aten_op;
    }

    return ops_.at(op_name);
  }

 private:
  ATenOperatorCache() = default;
  std::unordered_map<std::string, ATenOperator> ops_;
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
                        const std::vector<bool>& is_optional_arguments, std::vector<c10::IValue>& ivalue_arguments) {
  for (const auto& raw_argument : raw_arguments) {
    size_t index = raw_argument.first;
    TORCH_INTERNAL_ASSERT(index < ivalue_arguments.size());
    ivalue_arguments[index] = is_optional_arguments[index] ? c10::IValue(c10::optional<T>(raw_argument.second))
                                                           : c10::IValue(raw_argument.second);
  }
}

template <typename T>
void SetArrayIValueArguments(const std::vector<std::pair<size_t, std::vector<T>>>& raw_arguments,
                             const std::vector<bool>& is_optional_arguments,
                             std::vector<c10::IValue>& ivalue_arguments) {
  for (const auto& raw_argument : raw_arguments) {
    size_t index = raw_argument.first;
    TORCH_INTERNAL_ASSERT(index < ivalue_arguments.size());
    c10::List<T> list;
    for (T elem : raw_argument.second) {
      list.emplace_back(elem);
    }

    ivalue_arguments[index] =
        is_optional_arguments[index] ? c10::IValue(c10::optional<c10::List<T>>(list)) : c10::IValue(list);
  }
}

// TODO: Add more argument types, such as list type.
std::vector<DLManagedTensor*> ExecuteATenOperator(
    const char* op_name, const std::vector<std::pair<size_t, DLManagedTensor*>>& tensor_arguments,
    const std::vector<std::pair<size_t, int64_t>>& int_arguments,
    const std::vector<std::pair<size_t, float>>& float_arguments,
    const std::vector<std::pair<size_t, bool>>& bool_arguments,
    const std::vector<std::pair<size_t, std::vector<int64_t>>>& int_array_arguments,
    const std::vector<std::pair<size_t, std::vector<float>>>& float_array_arguments,
    const std::vector<std::pair<size_t, std::vector<bool>>>& bool_array_arguments) {
  std::string op_name_str(op_name);
  const auto& aten_op = ATenOperatorCache::Instance().GetOperator(op_name_str);

  // TODO: need to handle optional argument and arguments with default values.
  std::vector<c10::IValue> arguments;
  arguments.resize(aten_op.argument_size);
  for (const auto& tensor_argument : tensor_arguments) {
    size_t index = tensor_argument.first;
    at::Tensor tensor = at::fromDLPack(tensor_argument.second);
    bool has_transform_func = false;
    auto op_it = TENSOR_TRANSFORM_FUNCS.find(op_name_str);
    if (op_it != TENSOR_TRANSFORM_FUNCS.end()) {
      auto func_it = op_it->second.find(index);
      if (func_it != op_it->second.end()) {
        arguments[index] = func_it->second(tensor);
        has_transform_func = true;
      }
    }

    if (!has_transform_func) {
      arguments[index] =
          aten_op.is_optional_arguments[index] ? c10::IValue(c10::optional<at::Tensor>(tensor)) : c10::IValue(tensor);
    }
  }

  SetIValueArguments<int64_t>(int_arguments, aten_op.is_optional_arguments, arguments);
  SetIValueArguments<float>(float_arguments, aten_op.is_optional_arguments, arguments);
  SetIValueArguments<bool>(bool_arguments, aten_op.is_optional_arguments, arguments);
  SetArrayIValueArguments<int64_t>(int_array_arguments, aten_op.is_optional_arguments, arguments);
  SetArrayIValueArguments<float>(float_array_arguments, aten_op.is_optional_arguments, arguments);
  SetArrayIValueArguments<bool>(bool_array_arguments, aten_op.is_optional_arguments, arguments);

  torch::jit::Stack stack;
  for (size_t i = 0; i < arguments.size(); i++) {
    torch::jit::push(stack, arguments[i]);
  }

  aten_op.op->getOperation()(&stack);
  std::vector<DLManagedTensor*> result;
  for (const auto& ret : torch::jit::pop(stack, aten_op.return_size)) {
    result.emplace_back(at::toDLPack(ret.toTensor()));
  }

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
