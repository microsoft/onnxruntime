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
                        ATenOpBase<false>);

ONNX_OPERATOR_KERNEL_EX(ATenOpGrad, kMSDomain, 1, kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
                        ATenOpBase<true>);

template <bool is_backward>
ATenOpBase<is_backward>::ATenOpBase(const OpKernelInfo& info) : OpKernel(info) {
  std::string op_name;
  ORT_THROW_IF_ERROR(info.GetAttr("name", &op_name));
  const auto* op_config_ptr = aten_ops::GetATenOperatorConfig(op_name);
  ORT_ENFORCE(op_config_ptr, "ATen Op config for ", op_name, " is not found.");
  const auto& op_config = *op_config_ptr;
  op_name_ = is_backward ? op_config.backward_op_name : op_name;
  const auto& argument_configs = is_backward ? op_config.backward_argument_configs : op_config.forward_argument_configs;

  std::string custom_attributes_json = info.GetAttrOrDefault<std::string>("custom_attributes_json", "{}");
  aten_ops::AttributesJsonParser parser(custom_attributes_json);

  for (size_t i = 0; i < argument_configs.size(); i++) {
    const auto& argument_name = std::get<1>(argument_configs[i]);
    switch (std::get<0>(argument_configs[i])) {
      case aten_ops::TENSOR:
        tensor_argument_indices_.emplace_back(i);
        break;
      case aten_ops::INT:
        // JSON supports INT as 32-bit int, our attribute uses 64-bit int as INT type.
        int int_value;
        ORT_ENFORCE(parser.TryGetValue<int>(argument_name, int_value));
        int_arguments_.emplace_back(std::make_tuple(i, static_cast<int64_t>(int_value)));
        break;
      case aten_ops::FLOAT:
        float float_value;
        ORT_ENFORCE(parser.TryGetValue<float>(argument_name, float_value));
        float_arguments_.emplace_back(std::make_tuple(i, float_value));
        break;
      case aten_ops::BOOL:
        bool bool_value;
        ORT_ENFORCE(parser.TryGetValue<bool>(argument_name, bool_value));
        bool_arguments_.emplace_back(std::make_tuple(i, bool_value));
        break;
    }
  }
}

template <bool is_backward>
Status ATenOpBase<is_backward>::Compute(OpKernelContext* p_ctx) const {
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  std::vector<std::tuple<size_t, DLManagedTensor*>> tensor_arguments;
  for (size_t i = 0; i < tensor_argument_indices_.size(); i++) {
    OrtValue ort_value = *p_ctx_internal->GetInputMLValue(static_cast<int>(i));
    tensor_arguments.emplace_back(std::make_tuple(tensor_argument_indices_[i], dlpack::OrtValueToDlpack(ort_value)));
  }

  auto result = aten_ops::ATenOperatorExecutor::Instance()(op_name_, tensor_arguments, int_arguments_, float_arguments_,
                                                           bool_arguments_);

  for (size_t i = 0; i < result.size(); i++) {
    ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(static_cast<int>(i), dlpack::DlpackToOrtValue(result[i])));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
