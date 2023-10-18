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

int64_t countSubstrings(std::string_view str, std::string_view substr) {
  if (str.empty()) {
    return 0;
  }
  if (substr.empty()) {
    // Count consecutive whitespace as one delimiter
    int64_t count = 1;
    size_t pos = str.find_first_not_of(" ");
    while (pos != std::string::npos) {
      ++count;
      pos = str.find_first_not_of(" ", str.find_first_of(" ", pos));
    }
    return count;
  } else {
    int64_t count = 1;
    size_t pos = str.find(substr);
    while (pos != std::string::npos) {
      ++count;
      pos = str.find(substr, pos + substr.length());
    }
    return count;
  }
}

int64_t fillSubstrings(std::string_view str, std::string_view delimiter, gsl::span<std::string> output, int64_t output_index, size_t max_tokens) {
  if (str.empty()) {
    return 0;
  }
  if (delimiter.empty()) {
    // Count consecutive whitespace as one delimiter. Preceding and trailing whitespace is meant to be ignored.
    size_t pos = str.find_first_not_of(" ");
    size_t token_index = 0;
    while (token_index < max_tokens && pos != std::string::npos) {
      auto next_pos = token_index == max_tokens - 1 ? std::string::npos : str.find_first_of(" ", pos);
      output[output_index + token_index] = str.substr(pos, next_pos - pos);
      ++token_index;
      if (next_pos == std::string::npos) {
        break;
      }
      pos = str.find_first_not_of(" ", next_pos);
    }
    return token_index;
  } else {
    size_t pos = 0;
    size_t token_index = 0;
    while (token_index < max_tokens && pos != std::string::npos) {
      auto next_pos = token_index == max_tokens - 1 ? std::string::npos : str.find(delimiter, pos);
      output[output_index + token_index] = str.substr(pos, next_pos - pos);
      ++token_index;
      if (next_pos == std::string::npos) {
        break;
      }
      pos = next_pos + delimiter.size();
    }
    return token_index;
  }
}

StringSplit::StringSplit(const OpKernelInfo& info) : OpKernel(info) {
  info.GetAttrOrDefault("maxsplit", &maxsplit_, std::numeric_limits<int64_t>::max() - 1);  // TODO is this the right thing to do here?
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
  auto input_data = input->template DataAsSpan<std::string>();
  int64_t last_dim = 0;
  for (auto i = 0; i < input->Shape().Size(); ++i) {
    auto substring_count = countSubstrings(input_data[i], delimiter_);
    last_dim = std::max(last_dim, substring_count);
  }

  last_dim = std::min(last_dim, maxsplit_ + 1);

  // Set up num_substrings output
  auto num_substrings_data = num_substrings->template MutableDataAsSpan<int64_t>();
  // Set up splits output
  auto splits_shape = input->Shape().AsShapeVector();
  if (last_dim > 0) {
    splits_shape.push_back(last_dim);
  }
  auto splits_data = context->Output(0, splits_shape)->template MutableDataAsSpan<std::string>();
  auto splits_index = 0;
  for (auto i = 0; i < input->Shape().Size(); ++i) {
    num_substrings_data[i] = fillSubstrings(input_data[i], delimiter_, splits_data, splits_index, last_dim);
    splits_index += last_dim;
  }
  return Status::OK();
}

}  // namespace onnxruntime
