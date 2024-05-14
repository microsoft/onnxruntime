// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_split.h"
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

/// Calculate substrings in ``str`` delimited by ``delimiter``. A maximum of ``max_splits`` splits are permitted.
/// Returns a vector of string slices into ``str`` representing the substrings as string views. The user must ensure
/// the returned views' lifetime does not exceed ``str``'s.
void ComputeSubstrings(std::string_view str, std::string_view delimiter, int64_t max_splits, InlinedVector<std::string_view>& out) {
  if (str.empty()) {
    return;
  }
  if (delimiter.empty()) {
    // Count consecutive whitespace as one delimiter. Preceding and trailing whitespace is meant to be ignored.
    size_t pos = str.find_first_not_of(" ");
    int64_t token_count = 0;
    while (pos != std::string::npos) {
      if (token_count++ == max_splits) {
        // Trim down last substring as required in specification
        size_t next_pos = str.length() - 1;
        while (str[next_pos] == ' ') {
          next_pos--;
        }
        out.push_back(str.substr(pos, next_pos - pos + 1));
        break;
      } else {
        auto next_pos = str.find_first_of(" ", pos);
        out.push_back(str.substr(pos, next_pos - pos));
        pos = str.find_first_not_of(" ", next_pos);
      }
    }
  } else {
    size_t pos = 0;
    int64_t token_count = 0;
    while (pos != std::string::npos) {
      auto next_pos = str.find(delimiter, pos);
      if (token_count++ == max_splits || next_pos == std::string::npos) {
        out.push_back(str.substr(pos));
        break;
      }
      out.push_back(str.substr(pos, next_pos - pos));
      pos = next_pos + delimiter.size();
    }
  }
}

StringSplit::StringSplit(const OpKernelInfo& info) : OpKernel(info) {
  info.GetAttrOrDefault("maxsplit", &maxsplit_, std::numeric_limits<int64_t>::max() - 1);
  info.GetAttrOrDefault("delimiter", &delimiter_, std::string());
}

Status StringSplit::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  auto input_data = input->template DataAsSpan<std::string>();

  // Set up number of tokens output
  auto num_tokens_data = context->Output(1, input->Shape())->template MutableDataAsSpan<int64_t>();
  auto num_tokens_iter = num_tokens_data.begin();

  InlinedVector<InlinedVector<std::string_view>> input_slices;
  input_slices.reserve(input_data.size());
  size_t last_dim = 0;

  for (const auto& s : input_data) {
    auto& substrs = input_slices.emplace_back();
    ComputeSubstrings(s, delimiter_, maxsplit_, substrs);
    auto substr_count = substrs.size();
    last_dim = std::max(last_dim, substr_count);
    *num_tokens_iter = static_cast<int64_t>(substr_count);
    ++num_tokens_iter;
  }

  // Set up splits output
  auto splits_shape = input->Shape().AsShapeVector();
  splits_shape.push_back(last_dim);

  auto splits_data = context->Output(0, splits_shape)->template MutableDataAsSpan<std::string>();
  auto slices_iter = input_slices.begin();
  for (auto output_splits_iter = splits_data.begin(); output_splits_iter != splits_data.end(); output_splits_iter += last_dim, ++slices_iter) {
    std::copy(slices_iter->begin(), slices_iter->end(), output_splits_iter);
  }

  return Status::OK();
}

}  // namespace onnxruntime
