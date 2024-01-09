// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_split.h"
#include <algorithm>
#include "core/common/common.h"
namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(StringSplit, 20,
                         KernelDefBuilder()
                             .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
                             .TypeConstraint("T2", DataTypeImpl::GetTensorType<std::string>())
                             .TypeConstraint("T3", DataTypeImpl::GetTensorType<int64_t>()),
                         StringSplit);

/// Count the number of instances of substring ``substr`` in ``str``. If ``substr`` is an empty string it counts the
/// number of whitespace delimited words.
int64_t CountSubstrings(std::string_view str, std::string_view substr) {
  if (substr.empty()) {
    // Count consecutive whitespace as one delimiter
    int64_t count = 0;
    size_t pos = str.find_first_not_of(" ");
    while (pos != std::string::npos) {
      ++count;
      pos = str.find_first_not_of(" ", str.find_first_of(" ", pos));
    }
    return count;
  } else {
    int64_t count = 0;
    size_t pos = 0;
    while (pos != std::string::npos) {
      ++count;
      pos = str.find(substr, pos);
      if (pos != std::string::npos) {
        pos += substr.length();
      }
    }
    return count;
  }
}

/// Fill substrings of ``str`` based on split delimiter ``delimiter`` into ``output`` span. Restrict maximum number of
/// generated substrings to ``max_tokens``. The function returns the number of substrings generated (this is less or
/// equal to ``max_tokens``).
int64_t FillSubstrings(std::string_view str, std::string_view delimiter,
                       gsl::details::span_iterator<std::string> output, size_t max_tokens) {
  if (str.empty()) {
    return 0;
  }
  if (delimiter.empty()) {
    // Count consecutive whitespace as one delimiter. Preceding and trailing whitespace is meant to be ignored.
    size_t pos = str.find_first_not_of(" ");
    int64_t token_count = 0;
    while (pos != std::string::npos) {
      if (++token_count == max_tokens) {
        // trim down last substring as required in specification
        size_t next_pos = str.length() - 1;
        while (str[next_pos] == ' ') {
          next_pos--;
        }
        *output = str.substr(pos, next_pos - pos + 1);
        break;
      } else {
        auto next_pos = str.find_first_of(" ", pos);
        *output = str.substr(pos, next_pos - pos);
        pos = str.find_first_not_of(" ", next_pos);
      }

      output++;
    }
    return token_count;
  } else {
    size_t pos = 0;
    int64_t token_count = 0;
    while (pos != std::string::npos) {
      auto next_pos = token_count == max_tokens - 1 ? std::string::npos : str.find(delimiter, pos);
      *output++ = str.substr(pos, next_pos - pos);
      token_count++;
      if (next_pos == std::string::npos) {
        break;
      }
      pos = next_pos + delimiter.size();
    }
    return token_count;
  }
}

StringSplit::StringSplit(const OpKernelInfo& info) : OpKernel(info) {
  info.GetAttrOrDefault("maxsplit", &maxsplit_, std::numeric_limits<int64_t>::max() - 1);
  info.GetAttrOrDefault("delimiter", &delimiter_, std::string(""));
}

Status StringSplit::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  auto input_data = input->template DataAsSpan<std::string>();

  int64_t last_dim = 0;
  for (const auto& str : input_data) {
    last_dim = std::max(last_dim, CountSubstrings(str, delimiter_));
  }
  last_dim = std::min(last_dim, maxsplit_ + 1);

  // Set up splits output
  auto splits_shape = input->Shape().AsShapeVector();
  if (last_dim > 0) {
    splits_shape.push_back(last_dim);
  }
  auto splits_data = context->Output(0, splits_shape)->template MutableDataAsSpan<std::string>();
  auto output_splits_iter = splits_data.begin();

  // Set up number of tokens output
  auto* num_substrings = context->Output(1, input->Shape());
  auto num_substrings_data = num_substrings->template MutableDataAsSpan<int64_t>();
  auto output_num_tokens_iter = num_substrings_data.begin();

  for (auto input_iter = input_data.begin(); input_iter != input_data.end();
       input_iter++, output_splits_iter += last_dim, output_num_tokens_iter++) {
    *output_num_tokens_iter = FillSubstrings(*input_iter, delimiter_, output_splits_iter, last_dim);
  }
  return Status::OK();
}

}  // namespace onnxruntime
