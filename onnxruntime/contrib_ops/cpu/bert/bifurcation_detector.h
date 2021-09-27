// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
class BifurcationDetector : public OpKernel {
 public:
  explicit BifurcationDetector(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* src_tokens = context->Input<Tensor>(0);
    const Tensor* cur_tokens = context->Input<Tensor>(1);
    const Tensor* find_end_idx = context->Input<Tensor>(2);
    const Tensor* pred_tokens = context->Input<Tensor>(3);
    const auto* src_tokens_data = static_cast<const int64_t*>(src_tokens->DataRaw());
    const auto* cur_tokens_data = static_cast<const int64_t*>(cur_tokens->DataRaw());
    int64_t src_tokens_len = src_tokens->Shape().GetDims().at(0);
    int64_t cur_tokens_len = cur_tokens->Shape().GetDims().at(0);

    Tensor* out_tokens = nullptr;

    // Merge current tokens and predicted tokens.
    if (nullptr == pred_tokens) {
      out_tokens = context->Output(0, cur_tokens->Shape());
      auto* out_tokens_data = static_cast<int64_t*>(out_tokens->MutableDataRaw());
      memcpy(out_tokens_data, cur_tokens_data, cur_tokens_len * sizeof(int64_t));
    } else {
      const auto* pred_tokens_data = static_cast<const int64_t*>(pred_tokens->DataRaw());
      const int64_t find_end_idx_data = static_cast<const int64_t*>(find_end_idx->DataRaw())[0];
      int64_t pred_tokens_len = pred_tokens->Shape().GetDims().at(0);
      ORT_ENFORCE(src_tokens_len >= find_end_idx_data);
      ORT_ENFORCE(pred_tokens_len == (src_tokens_len + 1 - find_end_idx_data));
      int64_t idx = 0;
      for (; idx < src_tokens_len - find_end_idx_data; ++idx) {
        if (pred_tokens_data[idx] != src_tokens_data[idx + find_end_idx_data]) {
          break;
        }
      }
      // idx in [0, pred_tokens_len - 1]
      out_tokens = context->Output(0, TensorShape({cur_tokens_len + idx + 1}));
      auto* out_tokens_data = static_cast<int64_t*>(out_tokens->MutableDataRaw());
      memcpy(out_tokens_data, cur_tokens_data, cur_tokens_len * sizeof(int64_t));
      memcpy(out_tokens_data + cur_tokens_len, pred_tokens_data, (idx + 1) * sizeof(int64_t));
    }

    // Detect next bifurcation index.
    int64_t tokens_len = out_tokens->Shape().GetDims().at(0);
    int64_t min_gram = 1;
    int64_t max_gram = 3;
    int64_t idx = -1;
    const auto* tokens_data = static_cast<const int64_t*>(out_tokens->DataRaw());
    for (int64_t i = min_gram; i < max_gram + 1; ++i) {
      if (i > tokens_len) {
        break;
      }
      auto it = std::search(src_tokens_data, src_tokens_data + src_tokens_len, tokens_data + tokens_len - i, tokens_data + tokens_len);
      if (it == (src_tokens_data + src_tokens_len)) {
        break;
      } else {
        idx = it - src_tokens_data + i;
        if (idx >= src_tokens_len) {
          break;
        }
        auto it_2 = std::search(src_tokens_data + idx - i + 1, src_tokens_data + src_tokens_len, tokens_data + tokens_len - i, tokens_data + tokens_len);
        if (it_2 != (src_tokens_data + src_tokens_len)) {
          idx = -1;
          continue;
        }
      }
    }

    Tensor* out_end_idx = context->Output(1, find_end_idx->Shape());
    auto* out_end_idx_data = static_cast<int64_t*>(out_end_idx->MutableDataRaw());
    out_end_idx_data[0] = idx;

    return Status::OK();
  }
};
}  // namespace contrib
}  // namespace onnxruntime
