// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_attr_parser.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_executor.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_config.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(ATenOp, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        ATenOpForward);

ONNX_OPERATOR_KERNEL_EX(ATenOpGrad, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        ATenOpBackward);

template <typename T>
T GetAttributeValue(std::string argument_name, const aten_ops::AttributesJsonParser& parser,
                    const aten_ops::ATenOperatorConfig& op_config) {
  T value;
  if (!parser.TryGetValue<T>(argument_name, value)) {
    ORT_ENFORCE(op_config.TryGetDefaultValue<T>(argument_name, value), "Argument ", argument_name,
                " is not in attributes, and has no default value.");
  }

  return value;
}

template <typename T>
std::vector<T> GetArrayAttributeValue(std::string argument_name, const aten_ops::AttributesJsonParser& parser,
                                      const aten_ops::ATenOperatorConfig& op_config) {
  std::vector<T> list;
  if (!parser.TryGetArrayValue<T>(argument_name, list)) {
    list.clear();
    ORT_ENFORCE(op_config.TryGetDefaultArrayValue<T>(argument_name, list), "Argument ", argument_name,
                " is not in attributes, and has no default value.");
  }

  return list;
}

void ATenOpBase::Init(const OpKernelInfo& info, bool is_backward) {
  std::string op_name;
  ORT_THROW_IF_ERROR(info.GetAttr("name", &op_name));
  const auto* op_config_ptr = aten_ops::ATenOperatorConfigs::Instance().GetConfig(op_name);
  ORT_ENFORCE(op_config_ptr, "ATen Op config for ", op_name, " is not found.");
  const auto& op_config = *op_config_ptr;
  op_name_ = is_backward ? op_config.backward_op_name : op_name;
  const auto& argument_configs = is_backward ? op_config.backward_argument_configs : op_config.forward_argument_configs;

  std::string custom_attributes_json = info.GetAttrOrDefault<std::string>("custom_attributes_json", "{}");
  aten_ops::AttributesJsonParser parser(custom_attributes_json);

  for (size_t i = 0; i < argument_configs.size(); i++) {
    const auto& argument_name = std::get<1>(argument_configs[i]);
    switch (std::get<0>(argument_configs[i])) {
      case aten_ops::TENSOR: {
        tensor_argument_indices_.emplace_back(i);
      } break;
      case aten_ops::INT: {
        // JSON supports INT as 32-bit int, our attribute uses 64-bit int as INT type.
        int int_value = GetAttributeValue<int>(argument_name, parser, op_config);
        int_arguments_.emplace_back(std::make_pair(i, static_cast<int64_t>(int_value)));
      } break;
      case aten_ops::FLOAT: {
        float float_value = GetAttributeValue<float>(argument_name, parser, op_config);
        float_arguments_.emplace_back(std::make_pair(i, float_value));
      } break;
      case aten_ops::BOOL: {
        bool bool_value = GetAttributeValue<bool>(argument_name, parser, op_config);
        bool_arguments_.emplace_back(std::make_pair(i, bool_value));
      } break;
      case aten_ops::INT_ARRAY: {
        // JSON supports INT as 32-bit int, our attribute uses 64-bit int as INT type.
        std::vector<int> int_list = GetArrayAttributeValue<int>(argument_name, parser, op_config);
        std::vector<int64_t> long_list;
        for (int elem : int_list) {
          long_list.emplace_back(static_cast<int64_t>(elem));
        }
        int_array_arguments_.emplace_back(std::make_pair(i, long_list));
      } break;
      case aten_ops::FLOAT_ARRAY: {
        std::vector<float> float_list = GetArrayAttributeValue<float>(argument_name, parser, op_config);
        float_array_arguments_.emplace_back(std::make_pair(i, float_list));
      } break;
      case aten_ops::BOOL_ARRAY: {
        std::vector<bool> bool_list = GetArrayAttributeValue<bool>(argument_name, parser, op_config);
        bool_array_arguments_.emplace_back(std::make_pair(i, bool_list));
      } break;
      default:
        ORT_ENFORCE(false, "Not support for now.");
    }
  }
}

Status ATenOpBase::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  std::vector<std::pair<size_t, DLManagedTensor*>> tensor_arguments;
  for (size_t i = 0; i < tensor_argument_indices_.size(); i++) {
    OrtValue ort_value = *p_ctx_internal->GetInputMLValue(static_cast<int>(i));
    tensor_arguments.emplace_back(std::make_pair(tensor_argument_indices_[i], dlpack::OrtValueToDlpack(ort_value)));
  }

  auto result = aten_ops::ATenOperatorExecutor::Instance()(op_name_, tensor_arguments, int_arguments_, float_arguments_,
                                                           bool_arguments_, int_array_arguments_,
                                                           float_array_arguments_, bool_array_arguments_);

  for (size_t i = 0; i < result.size(); i++) {
    ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(result[i])));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
