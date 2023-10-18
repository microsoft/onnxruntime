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
  if (substr.empty()) {
    // Count consecutive whitespace as one delimiter
    bool in_whitespace = false;
    int64_t count = 1;
    for (const auto& c : str) {
      if (isspace(c)) {
        if (!in_whitespace) {
          in_whitespace = true;
          count += 1;
        }
      } else {
        in_whitespace = false;
      }
    }
    return count;
  } else {
    int64_t count = 1;
    size_t pos = 0;
    while ((pos = str.find(substr, pos)) != std::string::npos) {
      ++count;
      pos += substr.length();
    }
    return count;
  }
}

size_t fill_substrings(const std::string& str, const std::string& delimiter, gsl::span<std::string> output, int64_t output_index, size_t max_tokens) {
  if (delimiter.empty()) {
    // Count consecutive whitespace as one delimiter
    bool in_whitespace = false;
    size_t token_index = 0;
    size_t substr_start_index = 0;

    for (size_t i = 0; i < str.size(); ++i) {
      if (token_index == max_tokens - 1) {
        // if we are at the max_tokens-1 substring, the next and final substring should be the remainder of the string
        output[output_index + token_index] = str.substr(i);
        ++token_index;
        break;
      }
      if (!isspace(str[i])) {
        // if currently in a whitespace, this index marks the start of the current substring
        if (in_whitespace) {
          substr_start_index = i;
          in_whitespace = false;
        }
      } else if (!in_whitespace) {
        // if currently not in whitespace, this index is the end of a substring
        output[output_index + token_index] = str.substr(substr_start_index, i - substr_start_index);
        in_whitespace = true;
        ++token_index;
      }
    }
    return token_index;
  } else {
    size_t pos = 0;
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
  auto input_data = input->template DataAsSpan<std::string>();
  auto last_dim = maxsplit_;
  for (auto i = 0; i < input->Shape().Size(); ++i) {
    auto substring_count = countSubstrings(input_data[i], delimiter_);
    last_dim = std::max(last_dim, substring_count);
  }

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
    num_substrings_data[i] = static_cast<int64_t>(fill_substrings(input_data[i], delimiter_, splits_data, splits_index, last_dim));
    splits_index += last_dim;
  }
  return Status::OK();
}

}  // namespace onnxruntime
