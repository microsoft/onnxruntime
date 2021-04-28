// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/dlpack/dlpack_converter.h"
#include "core/framework/op_kernel_context_internal.h"
#include "orttraining/training_ops/cpu/aten_functions/aten_function_op.h"
#include "orttraining/training_ops/cpu/aten_functions/aten_function_utils.h"
#include "orttraining/training_ops/cpu/aten_functions/aten_function_config.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    ATenFunctionOp, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    ATenFunctionOpBase<false>);

ONNX_OPERATOR_KERNEL_EX(
    ATenFunctionOpGrad, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    ATenFunctionOpBase<true>);

template <bool is_backward>
ATenFunctionOpBase<is_backward>::ATenFunctionOpBase(const OpKernelInfo& info) : OpKernel(info) {
  std::string op_name;
  ORT_THROW_IF_ERROR(info.GetAttr("name", &op_name));
  ORT_ENFORCE(aten_functions::ATEN_FUNCTIONS.find(op_name) != aten_functions::ATEN_FUNCTIONS.end());
  aten_functions::ATenFunctionConfig fn_config = aten_functions::ATEN_FUNCTIONS.at(op_name);

  auto& ops = torch::jit::getAllOperatorsFor(
      torch::jit::Symbol::fromQualString(is_backward ? fn_config.backward_function_name : op_name));
  ORT_ENFORCE(ops.size() == 1);
  op_ = ops.front();

  std::string custom_attributes_json = info.GetAttrOrDefault<std::string>("custom_attributes_json", "{}");
  aten_functions::AttributesJsonParser parser(custom_attributes_json);

  size_t tensor_aurgument_index = 0;
  for (size_t i = 0; i < op_->schema().arguments().size(); i++) {
    c10::Argument argument = op_->schema().arguments()[i];
    if (is_backward && fn_config.custom_transformers.find(static_cast<int>(i)) != fn_config.custom_transformers.end()) {
      argument_configs_.emplace_back(std::make_tuple(TENSOR, tensor_aurgument_index++));
      transformers_[i] = fn_config.custom_transformers.at(static_cast<int>(i));
    } else if (argument.type()->kind() == c10::TypeKind::TensorType) {
      argument_configs_.emplace_back(std::make_tuple(TENSOR, tensor_aurgument_index++));
    } else {
      non_tensor_arguments_.emplace_back(parser.GetValue(argument));
      argument_configs_.emplace_back(std::make_tuple(NON_TENSOR, non_tensor_arguments_.size() - 1));
    }
  }
}

template <bool is_backward>
Status ATenFunctionOpBase<is_backward>::Compute(OpKernelContext* p_ctx) const {
  if (is_backward)
    std::cout << "Compute backward" << std::endl;
  else
    std::cout << "Compute forward" << std::endl;
  auto* p_ctx_internal = static_cast<OpKernelContextInternal*>(p_ctx);
  torch::jit::Stack stack;
  for (size_t i = 0; i < argument_configs_.size(); i++) {
    size_t index = std::get<1>(argument_configs_[i]);
    if (std::get<0>(argument_configs_[i]) == TENSOR) {
      OrtValue ort_value = *p_ctx_internal->GetInputMLValue(static_cast<int>(index));
      at::Tensor torch_tensor = aten_functions::ToTorchTensor(ort_value);
      if (transformers_.find(i) != transformers_.end()) {
        torch::jit::push(stack, transformers_.at(i)(torch_tensor));
      } else {
        torch::jit::push(stack, torch_tensor);
      }
    } else {
      torch::jit::push(stack, non_tensor_arguments_[index]);
    }
  }

  op_->getOperation()(&stack);
  // TODO: support single tensor as return value for now.
  at::Tensor output;
  torch::jit::pop(stack, output);
  ORT_RETURN_IF_ERROR(p_ctx_internal->SetOutputMLValue(0, aten_functions::FromTorchTensor(output)));
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
