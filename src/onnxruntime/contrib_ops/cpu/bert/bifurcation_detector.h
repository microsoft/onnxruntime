// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
class BifurcationDetector : public OpKernel {
 public:
  explicit BifurcationDetector(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("min_ngram_size", &min_ngram_size_).IsOK());
    ORT_ENFORCE(min_ngram_size_ > 0);
    ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK());
    ORT_ENFORCE(max_ngram_size_ > 0);
    ORT_ENFORCE(max_ngram_size_ >= min_ngram_size_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* src_tokens = context->Input<Tensor>(0);
    const Tensor* cur_tokens = context->Input<Tensor>(1);
    const Tensor* prev_suffix_match_idx = context->Input<Tensor>(2);
    const Tensor* pred_tokens = context->Input<Tensor>(3);
    const auto* src_tokens_data = static_cast<const int64_t*>(src_tokens->DataRaw());
    const auto* cur_tokens_data = static_cast<const int64_t*>(cur_tokens->DataRaw());
    int64_t src_tokens_len = src_tokens->Shape().GetDims()[0];
    int64_t cur_tokens_len = cur_tokens->Shape().GetDims()[0];

    Tensor* out_tokens = nullptr;

    // Find the bifurcation index of predicted tokens,
    // between source tokens, starting from previous suffix match index,
    // and predicted tokens.
    // Concat predicted tokens, starting from bifurcation index, to the back
    // of current tokens. This forms the output tokens.
    if (nullptr == pred_tokens) {
      // No prediction tokens. Output tokens equals to current tokens.
      out_tokens = context->Output(0, cur_tokens->Shape());
      auto* out_tokens_data = static_cast<int64_t*>(out_tokens->MutableDataRaw());
      memcpy(out_tokens_data, cur_tokens_data, SafeInt<size_t>(cur_tokens_len) * sizeof(int64_t));
    } else {
      const auto* pred_tokens_data = static_cast<const int64_t*>(pred_tokens->DataRaw());
      const int64_t prev_suffix_match_idx_data = static_cast<const int64_t*>(prev_suffix_match_idx->DataRaw())[0];
      int64_t pred_tokens_len = pred_tokens->Shape().GetDims()[0];
      // Find bifurcation index between prediction tokens, and source tokens
      // starting from previous suffix match index.
      ORT_ENFORCE(src_tokens_len >= prev_suffix_match_idx_data);
      ORT_ENFORCE(pred_tokens_len == (src_tokens_len + 1 - prev_suffix_match_idx_data));
      int64_t pred_bifur_idx = 0;
      for (; pred_bifur_idx < src_tokens_len - prev_suffix_match_idx_data; ++pred_bifur_idx) {
        if (pred_tokens_data[pred_bifur_idx] != src_tokens_data[pred_bifur_idx + prev_suffix_match_idx_data]) {
          break;
        }
      }
      // pred_bifur_idx in [0, pred_tokens_len - 1]
      out_tokens = context->Output(0, TensorShape({cur_tokens_len + pred_bifur_idx + 1}));
      auto* out_tokens_data = static_cast<int64_t*>(out_tokens->MutableDataRaw());
      memcpy(out_tokens_data, cur_tokens_data, SafeInt<size_t>(cur_tokens_len) * sizeof(int64_t));
      memcpy(out_tokens_data + cur_tokens_len, pred_tokens_data, SafeInt<size_t>(pred_bifur_idx + 1) * sizeof(int64_t));
    }

    // Detect suffix match index in source tokens, between source tokens and output tokens.
    // Detection is based on finding the appearances of last n-gram in output tokens
    // in source tokens.
    // A match is considered found if source tokens contain a single matching n-gram.
    // Return the index of the start of the n-gram in source tokens.
    // No matching if found if src tokens contain multiple or zero matching n-grams.
    // Return -1.
    int64_t tokens_len = out_tokens->Shape().GetDims()[0];
    int64_t min_gram = min_ngram_size_;
    int64_t max_gram = max_ngram_size_;
    int64_t suffix_idx = -1;
    const auto* tokens_data = static_cast<const int64_t*>(out_tokens->DataRaw());
    for (int64_t i = min_gram; i < max_gram + 1; ++i) {
      if (i > tokens_len) {
        break;
      }
      auto it = std::search(
          src_tokens_data,
          src_tokens_data + src_tokens_len,
          tokens_data + tokens_len - i,
          tokens_data + tokens_len);
      if (it == (src_tokens_data + src_tokens_len)) {
        break;
      } else {
        suffix_idx = it - src_tokens_data + i;
        if (suffix_idx >= src_tokens_len) {
          break;
        }
        auto it_2 = std::search(
            src_tokens_data + suffix_idx - i + 1,
            src_tokens_data + src_tokens_len,
            tokens_data + tokens_len - i,
            tokens_data + tokens_len);
        if (it_2 != (src_tokens_data + src_tokens_len)) {
          suffix_idx = -1;
          continue;
        }
      }
    }

    Tensor* out_suffix_match_idx = context->Output(1, prev_suffix_match_idx->Shape());
    auto* out_suffix_match_idx_data = static_cast<int64_t*>(out_suffix_match_idx->MutableDataRaw());
    out_suffix_match_idx_data[0] = suffix_idx;

    return Status::OK();
  }

 private:
  int64_t min_ngram_size_;
  int64_t max_ngram_size_;
};
}  // namespace contrib
}  // namespace onnxruntime
