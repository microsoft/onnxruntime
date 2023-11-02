// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/regex_full_match.h"
#include "core/common/common.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    RegexFullMatch,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    RegexFullMatch);

RegexFullMatch::RegexFullMatch(const OpKernelInfo& info) : OpKernel(info), re_{info.GetAttr<std::string>("pattern")} {
  ORT_ENFORCE(re_.ok(), "Invalid regex pattern: ", re_.pattern());
}

Status RegexFullMatch::Compute(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto input_data = input_tensor->template DataAsSpan<std::string>();
  auto* output_tensor = context->Output(0, input_tensor->Shape());
  auto output_data = output_tensor->template MutableDataAsSpan<bool>();
  auto output_iter = output_data.begin();
  auto input_iter = input_data.begin();
  while (input_iter != input_data.end()) {
    *output_iter = RE2::FullMatch(*input_iter, re_);
    input_iter++;
    output_iter++;
  }
  return Status::OK();
}

}  // namespace onnxruntime
