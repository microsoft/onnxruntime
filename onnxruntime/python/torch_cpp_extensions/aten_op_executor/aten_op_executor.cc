// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/torch.h>
#include <ATen/DLConvertor.h>
#include <unordered_map>
#include <vector>

template <typename TSrc, typename TDst>
c10::IValue ToIValue(const DLManagedTensor* dlpack, bool is_optional) {
  TORCH_INTERNAL_ASSERT((dlpack->dl_tensor.ndim == 0 && dlpack->dl_tensor.shape == nullptr) ||
                        (dlpack->dl_tensor.ndim == 1 && dlpack->dl_tensor.shape[0] == 1));
  TDst value = static_cast<TDst>(*reinterpret_cast<const TSrc*>(dlpack->dl_tensor.data));
  return is_optional ? c10::IValue(c10::optional<TDst>(value)) : c10::IValue(value);
}

template <typename TSrc, typename TDst>
c10::IValue ToListIValue(const DLManagedTensor* dlpack, bool is_optional) {
  TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.ndim == 1);
  const TSrc* p_data = reinterpret_cast<const TSrc*>(dlpack->dl_tensor.data);
  c10::List<TDst> list_value;
  size_t len = static_cast<size_t>(dlpack->dl_tensor.shape[0]);
  for (size_t i = 0; i < len; i++) {
    list_value.emplace_back(static_cast<TDst>(p_data[i]));
  }
  return is_optional ? c10::IValue(c10::optional<c10::List<TDst>>(list_value)) : c10::IValue(list_value);
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
    // Create the torch tensor from this DLPack no matter we need it or not below,
    // so that the dlpack's deleter will be triggered when torch tensor is out of scope.
    at::Tensor tensor = at::fromDLPack(dlpack);
    switch (elem_kinds[index]) {
      case c10::TypeKind::TensorType: {
        i_value = is_optional ? c10::IValue(c10::optional<at::Tensor>(tensor)) : c10::IValue(tensor);
      } break;
      case c10::TypeKind::IntType: {
        TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLInt &&
                              dlpack->dl_tensor.dtype.bits == 64);
        i_value = is_list ? ToListIValue<int64_t, int64_t>(dlpack, is_optional)
                          : ToIValue<int64_t, int64_t>(dlpack, is_optional);
      } break;
      case c10::TypeKind::FloatType: {
        TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLFloat &&
                              dlpack->dl_tensor.dtype.bits == 32);
        // PyTorch's IValue doesn't support float, so we convert it to double.
        i_value =
            is_list ? ToListIValue<float, double>(dlpack, is_optional) : ToIValue<float, double>(dlpack, is_optional);
      } break;
      case c10::TypeKind::BoolType: {
        // In torch 1.8.1, exporter has bug which exports bool constant to int64 type tensor.
        // This bug has been fixed since torch 1.9.0. To make torch 1.8.1 work, add special handling here.
        if (dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLInt && dlpack->dl_tensor.dtype.bits == 64) {
          i_value =
              is_list ? ToListIValue<int64_t, bool>(dlpack, is_optional) : ToIValue<int64_t, bool>(dlpack, is_optional);
        } else {
          TORCH_INTERNAL_ASSERT(dlpack->dl_tensor.dtype.code == DLDataTypeCode::kDLUInt &&
                                dlpack->dl_tensor.dtype.bits == 8);
          i_value = is_list ? ToListIValue<bool, bool>(dlpack, is_optional) : ToIValue<bool, bool>(dlpack, is_optional);
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

  const ATenOperator& GetOperator(const std::string& op_name, const std::string& overload_name) {
    // PyTorch ONNX converter creates ATen operators with name without domain
    std::string final_op_name = op_name;
    auto pos = op_name.find("::");
    if (pos == std::string::npos) {
      final_op_name = std::string("aten::" + op_name);
    }

    auto key = std::make_pair(final_op_name, overload_name);
    if (ops_.find(key) == ops_.end()) {
      c10::OperatorName full_name(final_op_name, overload_name);
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

const std::unordered_map<std::string, std::unordered_set<size_t>> kCpuTensorInputsMap = {
    {"_efficient_attention_forward", {4, 5, 11, 12}}, {"_efficient_attention_backward", {6, 7, 12, 13}}};

const std::unordered_map<std::string, std::unordered_set<size_t>> kCpuTensorOutputsMap = {
    {"_efficient_attention_forward", {2, 3}}};

// Backend uses this function to check if an argument is CPU input or not.
bool IsCpuArgument(const char* op_name, const char* overload_name, size_t index, bool is_input) {
  if (is_input) {
    // If the argument is non-tensor type, it's CPU argument.
    const auto& aten_op = ATenOperatorCache::Instance().GetOperator(op_name, overload_name);
    TORCH_INTERNAL_ASSERT(index < aten_op.argument_size);
    if (aten_op.elem_kinds[index] != c10::TypeKind::TensorType) {
      return true;
    }
  }

  std::string full_name = std::string(op_name);
  std::string overload_name_str = std::string(overload_name);
  if (overload_name_str != "") {
    full_name += ("." + overload_name_str);
  }

  const auto& cpu_tensors_map = is_input ? kCpuTensorInputsMap : kCpuTensorOutputsMap;
  return cpu_tensors_map.find(full_name) != cpu_tensors_map.end() &&
         cpu_tensors_map.at(full_name).find(index) != cpu_tensors_map.at(full_name).end();
}

void ExecuteATenOperator(const char* op_name, const char* overload_name, size_t input_size,
                         DLManagedTensor** dlpack_inputs, size_t output_size, DLManagedTensor** dlpack_outputs) {
  const auto& aten_op = ATenOperatorCache::Instance().GetOperator(op_name, overload_name);
  TORCH_INTERNAL_ASSERT(input_size == aten_op.argument_size);
  std::vector<c10::IValue> arguments;
  for (size_t i = 0; i < input_size; ++i) {
    arguments.emplace_back(aten_op.ToIValueArgument(dlpack_inputs[i], i));
  }

  torch::jit::Stack stack;
  for (size_t i = 0; i < arguments.size(); i++) {
    torch::jit::push(stack, arguments[i]);
  }

#ifndef TORCH_VERSION_PREEQ
#define TORCH_VERSION_PREEQ(x, y) \
  ((TORCH_VERSION_MAJOR == (x) && TORCH_VERSION_MINOR >= (y)) || (TORCH_VERSION_MAJOR > (x)))
#endif

// pull request https://github.com/pytorch/pytorch/pull/63414 introduced
// a backwards incompatibility by changing the API. To make ORTModule
// work with both torch versions >=1.10 as well as < 1.10, we need
// preprocessor checks
#if TORCH_VERSION_PREEQ(1, 10)
  // torch version is >= 1.10
  aten_op.op->getOperation()(stack);
#else
  // torch version is < 1.10
  aten_op.op->getOperation()(&stack);
#endif

  TORCH_INTERNAL_ASSERT(output_size == aten_op.return_size);
  size_t output_index = 0;
  for (const auto& ret : torch::jit::pop(stack, output_size)) {
    const auto& tensor = ret.toTensor();
    dlpack_outputs[output_index++] =
        tensor.defined() ? at::toDLPack(tensor.is_contiguous() ? tensor : tensor.contiguous()) : nullptr;
  }
}

size_t is_cpu_argument_address() { return reinterpret_cast<size_t>(&IsCpuArgument); }
size_t execute_aten_operator_address() { return reinterpret_cast<size_t>(&ExecuteATenOperator); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_cpu_argument_address", &is_cpu_argument_address, "Address of tensor argument check.");
  m.def("execute_aten_operator_address", &execute_aten_operator_address, "Address of Aten operator executor");
}
