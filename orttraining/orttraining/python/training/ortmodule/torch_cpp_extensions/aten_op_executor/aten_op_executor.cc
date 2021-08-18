// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <unordered_map>
#include <vector>

template <typename T>
c10::IValue ToIValue(const DLManagedTensor* dlpack, bool is_optional) {
  TORCH_INTERNAL_ASSERT((dlpack->dl_tensor.ndim == 0 && dlpack->dl_tensor.shape == nullptr) ||
                        (dlpack->dl_tensor.ndim == 1 && dlpack->dl_tensor.shape[0] == 1));
  T value = *reinterpret_cast<const T*>(dlpack->dl_tensor.data);
  return is_optional ? c10::IValue(c10::optional<T>(value)) : c10::IValue(value);
}

template <typename T>
c10::IValue ToListIValue(const DLManagedTensor* dlpack, bool is_optional) {
  TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.ndim == 1);
  const T* p_data = reinterpret_cast<const T*>(dlpack->dl_tensor.data);
  c10::List<T> list_value;
  size_t len = static_cast<size_t>(dlpack->dl_tensor.shape[0]);
  for (size_t i = 0; i < len; i++) {
    list_value.emplace_back(p_data[i]);
  }
  return is_optional ? c10::IValue(c10::optional<c10::List<T>>(list_value)) : c10::IValue(list_value);
}

c10::IValue Int64ToBoolIValue(const DLManagedTensor* dlpack, bool is_list, bool is_optional) {
  if (is_list) {
    TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.ndim == 1);
    const int64_t* p_data = reinterpret_cast<const int64_t*>(dlpack->dl_tensor.data);
    c10::List<bool> list_value;
    size_t len = static_cast<size_t>(dlpack->dl_tensor.shape[0]);
    for (size_t i = 0; i < len; i++) {
      list_value.emplace_back(static_cast<bool>(p_data[i]));
    }
    return is_optional ? c10::IValue(c10::optional<c10::List<bool>>(list_value)) : c10::IValue(list_value);
  }

  TORCH_INTERNAL_ASSERT((dlpack->dl_tensor.ndim == 0 && dlpack->dl_tensor.shape == nullptr) ||
                        (dlpack->dl_tensor.ndim == 1 && dlpack->dl_tensor.shape[0] == 1));
  bool value = static_cast<bool>(*reinterpret_cast<const int64_t*>(dlpack->dl_tensor.data));
  return is_optional ? c10::IValue(c10::optional<bool>(value)) : c10::IValue(value);
}

struct ATenOperator {
  std::shared_ptr<torch::jit::Operator> op;
  size_t argument_size;
  std::vector<c10::TypeKind> elem_kinds;
  std::vector<bool> is_list_arguments;
  std::vector<bool> is_optional_arguments;
  std::vector<c10::optional<c10::IValue>> default_values;
  size_t return_size;

  c10::IValue ToIValueArgument(const DLManagedTensor* dlpack, size_t index) const {
    TORCH_INTERNAL_ASSERT(index < argument_size);
    bool is_optional = is_optional_arguments[index];
    TORCH_INTERNAL_ASSERT(dlpack || is_optional || default_values[index]);
    if (!dlpack) {
      if (is_optional) {
        // Optional argument always has no default value.
        return c10::IValue(c10::nullopt);
      }

      return *default_values[index];
    }

    bool is_list = is_list_arguments[index];
    c10::IValue i_value;
    switch (elem_kinds[index]) {
      case c10::TypeKind::TensorType: {
        at::Tensor tensor = at::fromDLPack(dlpack);
        i_value = is_optional ? c10::IValue(c10::optional<at::Tensor>(tensor)) : c10::IValue(tensor);
      } break;
      case c10::TypeKind::IntType: {
        TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLInt &&
                              dlpack->dl_tensor.dtype.bits == 64);
        i_value = is_list ? ToListIValue<int64_t>(dlpack, is_optional) : ToIValue<int64_t>(dlpack, is_optional);
      } break;
      case c10::TypeKind::FloatType: {
        TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLFloat &&
                              dlpack->dl_tensor.dtype.bits == 32);
        i_value = is_list ? ToListIValue<float>(dlpack, is_optional) : ToIValue<float>(dlpack, is_optional);
      } break;
      case c10::TypeKind::BoolType: {
        // In torch 1.8.1, exporter has bug which exports bool constant to int64 type tensor.
        // This bug has been fixed since torch 1.9.0. To make torch 1.8.1 work, add special handling here.
        if (dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLInt && dlpack->dl_tensor.dtype.bits == 64) {
          i_value = Int64ToBoolIValue(dlpack, is_list, is_optional);
        } else {
          TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLUInt &&
                                dlpack->dl_tensor.dtype.bits == 8);
          i_value = is_list ? ToListIValue<bool>(dlpack, is_optional) : ToIValue<bool>(dlpack, is_optional);
        }
      } break;
      default:  // TODO: will add more type support if needed.
        TORCH_INTERNAL_ASSERT(false);
    }

    return i_value;
  }
};

struct PairHash {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

class ATenOperatorCache {
 public:
  static ATenOperatorCache& Instance() {
    static ATenOperatorCache instance;
    return instance;
  }

  // TODO: add lock. Currently we are executing nodes sequentially, so lock is not needed.
  const ATenOperator& GetOperator(const std::string& op_name, const std::string& overload_name) {
    auto key = std::make_pair(op_name, overload_name);
    if (ops_.find(key) == ops_.end()) {
      c10::OperatorName full_name(op_name, overload_name);
      auto op = torch::jit::findOperatorFor(full_name);
      TORCH_INTERNAL_ASSERT(op);
      ATenOperator aten_op;
      aten_op.op = op;
      const auto& schema = aten_op.op->schema();
      aten_op.argument_size = schema.arguments().size();
      for (const auto& argument : schema.arguments()) {
        c10::TypePtr type = argument.type();
        c10::TypeKind elem_type = type->kind();
        bool is_optional = elem_type == c10::TypeKind::OptionalType;
        bool is_list = elem_type == c10::TypeKind::ListType;
        if (is_optional) {
          type = reinterpret_cast<c10::OptionalType*>(type.get())->getElementType();
          elem_type = type->kind();
          is_list = elem_type == c10::TypeKind::ListType;
        }
        if (is_list) {
          elem_type = reinterpret_cast<c10::ListType*>(type.get())->getElementType()->kind();
        }
        TORCH_INTERNAL_ASSERT(elem_type != c10::TypeKind::TensorType || !is_list);
        aten_op.elem_kinds.emplace_back(elem_type);
        aten_op.is_list_arguments.emplace_back(is_list);
        aten_op.is_optional_arguments.emplace_back(is_optional);
        aten_op.default_values.emplace_back(argument.default_value());
      }
      aten_op.return_size = schema.returns().size();
      for (const auto& ret : schema.returns()) {
        TORCH_INTERNAL_ASSERT(ret.type()->kind() == c10::TypeKind::TensorType);
      }
      ops_.emplace(key, aten_op);
    }
    return ops_.at(key);
  }

 private:
  ATenOperatorCache() = default;
  std::unordered_map<std::pair<std::string, std::string>, ATenOperator, PairHash> ops_;
};

// AutogradContext saves forward inputs that require grad, and the output tensor.
// Backward executor will call output.backward, and fetch input grads from input tensors.
struct AutogradContext {
  std::vector<c10::IValue> inputs;
  std::vector<bool> is_optional_inputs;
  c10::IValue output;
};

class AutogradContextCache {
 public:
  static AutogradContextCache& Instance() {
    static AutogradContextCache instance;
    return instance;
  }

  // TODO: add lock. Currently we are executing nodes sequentially, so lock is not needed.
  int64_t Insert(const AutogradContext& autograd_context) {
    int64_t context_id = CreateId();
    autograd_contexts_.emplace(context_id, autograd_context);
    return context_id;
  }

  AutogradContext Pop(int64_t context_id) {
    auto it = autograd_contexts_.find(context_id);
    TORCH_INTERNAL_ASSERT(it != autograd_contexts_.end());
    AutogradContext autograd_context = it->second;
    autograd_contexts_.erase(it);
    return autograd_context;
  }

 private:
  static int64_t CreateId() {
    static std::atomic<int64_t> context_id_{0};
    return context_id_++;
  }

  AutogradContextCache() = default;
  std::unordered_map<int64_t, AutogradContext> autograd_contexts_;
};

// Backend uses this function to check if an argument is CPU input (non-tensor argument) or not.
bool IsTensorArgument(const char* op_name, const char* overload_name, size_t index) {
  const auto& aten_op = ATenOperatorCache::Instance().GetOperator(op_name, overload_name);
  TORCH_INTERNAL_ASSERT(index < aten_op.argument_size);
  return aten_op.elem_kinds[index] == c10::TypeKind::TensorType;
}

c10::IValue ExecuteInternal(const std::shared_ptr<torch::jit::Operator>& op, const std::vector<c10::IValue>& arguments,
                            size_t return_size, std::vector<DLManagedTensor*>& result) {
  torch::jit::Stack stack;
  for (size_t i = 0; i < arguments.size(); i++) {
    torch::jit::push(stack, arguments[i]);
  }

  op->getOperation()(&stack);
  c10::IValue first_output;
  bool is_first = true;
  for (const auto& ret : torch::jit::pop(stack, return_size)) {
    if (is_first) {
      first_output = ret;
      is_first = false;
    }
    const auto& tensor = ret.toTensor();
    result.emplace_back(at::toDLPack(tensor.is_contiguous() ? tensor : tensor.contiguous()));
  }

  return first_output;
}

std::vector<DLManagedTensor*> ExecuteATenOperator(const char* op_name, const char* overload_name,
                                                  const std::vector<DLManagedTensor*>& dlpacks,
                                                  const std::vector<int64_t>& requires_grad, int64_t* p_context_id) {
  const auto& aten_op = ATenOperatorCache::Instance().GetOperator(op_name, overload_name);
  TORCH_INTERNAL_ASSERT(dlpacks.size() == aten_op.argument_size);
  std::vector<c10::IValue> arguments;
  for (size_t i = 0; i < dlpacks.size(); i++) {
    arguments.emplace_back(aten_op.ToIValueArgument(dlpacks[i], i));
  }

  AutogradContext autograd_context;
  for (size_t i = 0; i < requires_grad.size(); i++) {
    bool is_optional = aten_op.is_optional_arguments[i];
    if (is_optional) {
      arguments[i].toOptional<at::Tensor>().value().requires_grad_(true);
    } else {
      arguments[i].toTensor().requires_grad_(true);
    }

    autograd_context.inputs.emplace_back(arguments[i]);
    autograd_context.is_optional_inputs.emplace_back(is_optional);
  }

  TORCH_INTERNAL_ASSERT(p_context_id || requires_grad.empty());
  std::vector<DLManagedTensor*> result;
  if (p_context_id) {
    // We need to enable Autograd in case there is not.
    c10::AutoGradMode auto_grad_mode(true);
    autograd_context.output = ExecuteInternal(aten_op.op, arguments, aten_op.return_size, result);
    *p_context_id = AutogradContextCache::Instance().Insert(autograd_context);
  } else {
    ExecuteInternal(aten_op.op, arguments, aten_op.return_size, result);
  }

  return result;
}

std::vector<DLManagedTensor*> ExecuteATenOpBackward(DLManagedTensor* dlpack, int64_t context_id) {
  // Add this according to the PyGILState_Check in torch/csrc/autograd/python_engine.cpp.
  // Otherwise, the autograd engine will not work.
  pybind11::gil_scoped_release no_gil;
  AutogradContext autograd_context = AutogradContextCache::Instance().Pop(context_id);
  autograd_context.output.toTensor().backward(at::fromDLPack(dlpack));
  std::vector<DLManagedTensor*> result;
  for (size_t i = 0; i < autograd_context.inputs.size(); i++) {
    at::Tensor intput_grad = autograd_context.is_optional_inputs[i]
                                 ? autograd_context.inputs[i].toOptional<at::Tensor>().value().grad()
                                 : autograd_context.inputs[i].toTensor().grad();
    result.emplace_back(at::toDLPack(intput_grad.is_contiguous() ? intput_grad : intput_grad.contiguous()));
  }

  return result;
}

size_t is_tensor_argument_address() { return reinterpret_cast<size_t>(&IsTensorArgument); }
size_t execute_aten_operator_address() { return reinterpret_cast<size_t>(&ExecuteATenOperator); }
size_t execute_aten_op_backward_address() { return reinterpret_cast<size_t>(&ExecuteATenOpBackward); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_tensor_argument_address", &is_tensor_argument_address, "Address of tensor argument check.");
  m.def("execute_aten_operator_address", &execute_aten_operator_address, "Address of Aten operator executor");
  m.def("execute_aten_op_backward_address", &execute_aten_op_backward_address, "Address of Aten op backward executor");
}
