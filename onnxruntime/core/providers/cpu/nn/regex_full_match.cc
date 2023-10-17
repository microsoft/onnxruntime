#include "regex_full_match.h"
#include "core/common/common.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    RegexFullMatch,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),
    RegexFullMatch);

RegexFullMatch::RegexFullMatch(const OpKernelInfo& info) : OpKernel(info) {
  ORT_ENFORCE(info.GetAttr<std::string>("pattern", &pattern_).IsOK());
  ORT_ENFORCE(RE2(pattern_).ok(), "Invalid pattern: ", pattern_);
}

Status RegexFullMatch::Compute(OpKernelContext* context) const {
  RE2 re(pattern_);
  const auto* input_tensor = context->Input<Tensor>(0);
  if (nullptr == input_tensor) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Input count mismatch");
  }
  auto* output_tensor = context->Output(0, input_tensor->Shape());
  if (nullptr == output_tensor) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output count mismatch");
  }
  const auto input_data = input_tensor->template DataAsSpan<std::string>();
  auto output_data = output_tensor->template MutableDataAsSpan<bool>();
  const auto N = input_tensor->Shape().Size();
  auto output_iter = output_data.begin();
  auto input_iter = input_data.begin();
  while (input_iter != output_data.end()) {
    *output_iter = RE2::FullMatch(*input_iter, re);
    input_iter++;
    output_iter++;
  }
  return Status::OK();
}

}  // namespace onnxruntime
