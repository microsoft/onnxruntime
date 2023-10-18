#include "string_split.h"
#include "core/common/common.h"
#include <algorithm>
namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    StringSplit,
    20,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int64_t>()),
    StringSplit);

int64_t countSubstrings(const std::string& str, const std::string& substr) {
  if (str.empty()) {
    return 0;
  }
  int64_t count = 1;
  size_t pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos) {
    ++count;
    pos += substr.length();
  }

  return count;
}

size_t fill_substrings(const std::string& str, const std::string& delimiter, gsl::span<std::string> output, int64_t output_index, int64_t max_tokens) {
  // Fill output with substrings of str, delimited by delimiter and place into output starting at output_index and incrementing up.
  // Up to max_tokens substrings should be reached. If we are done before max_tokens, fill the rest with "". If we would not be done after max_tokens, make sure the max_tokensth substring is the remainder of the string.
  auto pos = 0;
  size_t token_index = 0;
  while (token_index < max_tokens) {
    auto next_pos = str.find(delimiter, pos);
    output[output_index + token_index] = str.substr(pos, next_pos - pos);
    pos = next_pos + delimiter.size();
    ++token_index;
    if (next_pos == std::string::npos) {
      break;
    }
  }
  return token_index;
}

StringSplit::StringSplit(const OpKernelInfo& info) : OpKernel(info) {
  info.GetAttrOrDefault("maxsplit", &maxsplit_, static_cast<int64_t>(-1));  // TODO is this the right thing to do here?
  info.GetAttrOrDefault("delimiter", &delimiter_, std::string(""));
}

Status StringSplit::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  if (nullptr == input) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  }
  auto* num_substrings = context->Output(1, input->Shape());
  if (nullptr == num_substrings) {
    return Status(common::ONNXRUNTIME, common::FAIL, "output count mismatch");
  }
  if ("" == delimiter_) {
    // TODO: takes consecutive whitespace as delimiter
  } else {
    auto input_data = input->template DataAsSpan<std::string>();
    auto last_dim = maxsplit_;
    for (auto i = 0; i < input->Shape().Size(); ++i) {
      auto substring_count = countSubstrings(input_data[i], delimiter_);
      last_dim = std::max(last_dim, substring_count);
    }
    // 1. instantiate output shape to be shape input + (last_dim,)
    // 2. maintain output tensor index/pointer. Iterate over input tensor; split tensor into last_dim substrings (with "" at end for extra); copy into output tensor and output pointer/index. advance output pointer/index by last_dim.

    // Set up num_substrings output
    auto num_substrings_data = num_substrings->template MutableDataAsSpan<int64_t>();
    // Set up splits output
    auto splits_shape = input->Shape().AsShapeVector();
    splits_shape.push_back(last_dim);
    auto splits_data = context->Output(0, splits_shape)->template MutableDataAsSpan<std::string>();
    auto splits_index = 0;
    for (auto i = 0; i < input->Shape().Size(); ++i) {
      num_substrings_data[i] = static_cast<int64_t>(fill_substrings(input_data[i], delimiter_, splits_data, splits_index, last_dim));
      splits_index += last_dim;
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
