// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/string_split.h"
#include <algorithm>
#include <limits>
#include <string>
#include "core/common/common.h"
namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(StringSplit, 20,
                         KernelDefBuilder()
                             .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
                             .TypeConstraint("T2", DataTypeImpl::GetTensorType<std::string>())
                             .TypeConstraint("T3", DataTypeImpl::GetTensorType<int64_t>()),
                         StringSplit);

/// Fill substrings of ``str`` based on split delimiter ``delimiter`` into ``output`` span. Restrict maximum number of
/// generated substrings to ``max_tokens``. The function returns the number of substrings generated (this is less or
/// equal to ``max_tokens``).
InlinedVector<std::string_view> FillSubstrings(std::string_view str, std::string_view delimiter, int64_t max_splits) {
  InlinedVector<std::string_view> output;
  if (str.empty()) {
    return output;
  }
  if (delimiter.empty()) {
    // Count consecutive whitespace as one delimiter. Preceding and trailing whitespace is meant to be ignored.
    size_t pos = str.find_first_not_of(" ");
    int64_t token_count = 0;
    while (pos != std::string::npos) {
      if (token_count++ == max_splits) {
        // trim down last substring as required in specification
        size_t next_pos = str.length() - 1;
        while (str[next_pos] == ' ') {
          next_pos--;
        }
        output.push_back(str.substr(pos, next_pos - pos + 1));
        break;
      } else {
        auto next_pos = str.find_first_of(" ", pos);
        output.push_back(str.substr(pos, next_pos - pos));
        pos = str.find_first_not_of(" ", next_pos);
      }
    }
    return output;
  } else {
    size_t pos = 0;
    int64_t token_count = 0;
    while (pos != std::string::npos) {
      auto next_pos = str.find(delimiter, pos);
      if (token_count++ == max_splits || next_pos == std::string::npos) {
        output.push_back(str.substr(pos));
        break;
      }
      output.push_back(str.substr(pos, next_pos - pos));
      pos = next_pos + delimiter.size();
    }
    return output;
  }
}

StringSplit::StringSplit(const OpKernelInfo& info) : OpKernel(info) {
  info.GetAttrOrDefault("maxsplit", &maxsplit_, std::numeric_limits<int64_t>::max() - 1);
  info.GetAttrOrDefault("delimiter", &delimiter_, std::string(""));
}

Status StringSplit::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  auto input_data = input->template DataAsSpan<std::string>();

  // Set up number of tokens output
  auto num_tokens_data = context->Output(1, input->Shape())->template MutableDataAsSpan<int64_t>();
  auto num_tokens_iter = num_tokens_data.begin();

  int64_t last_dim = 1;

  InlinedVector<InlinedVector<std::string_view>> input_slices;
  input_slices.reserve(input_data.size());
  auto input_slice_iterator = input_slices.begin();
  for (auto input_iter = input_data.begin(); input_iter != input_data.end(); input_iter++, input_slice_iterator++, num_tokens_iter++) {
    auto substrs = FillSubstrings(*input_iter, delimiter_, maxsplit_);
    auto substr_count = static_cast<int64_t>(substrs.size());
    input_slices.push_back(substrs);
    last_dim = std::max(last_dim, substr_count);
    *num_tokens_iter = substr_count;
  }

  last_dim = std::min(last_dim, maxsplit_ + 1);

  // Set up splits output
  auto splits_shape = input->Shape().AsShapeVector();
  splits_shape.push_back(last_dim);

  auto splits_data = context->Output(0, splits_shape)->template MutableDataAsSpan<std::string>();
  auto slices_iter = input_slices.begin();
  for (auto output_splits_iter = splits_data.begin(); output_splits_iter != splits_data.end(); output_splits_iter += last_dim, slices_iter++) {
    const auto output_slices = *slices_iter;
    std::copy(output_slices.begin(), output_slices.end(), output_splits_iter);
  }

  return Status::OK();
}

}  // namespace onnxruntime
