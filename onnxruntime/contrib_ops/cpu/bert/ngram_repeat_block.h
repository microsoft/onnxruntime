// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {
class NGramRepeatBlock : public OpKernel {
 public:
  explicit NGramRepeatBlock(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("ngram_size", &ngram_size_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* input_ids = context->Input<Tensor>(0);
    const Tensor* scores = context->Input<Tensor>(1);
    Tensor* output = context->Output(0, scores->Shape());

    const auto* scores_source = static_cast<const float*>(scores->DataRaw());
    auto* scores_target = static_cast<float*>(output->MutableDataRaw());
    if (scores_source != scores_target) {
      memcpy(scores_target, scores_source, scores->Shape().Size() * sizeof(float));
    }

    const auto& input_ids_dims = input_ids->Shape().GetDims();
    const auto& scores_dims = scores->Shape().GetDims();
    ORT_ENFORCE(input_ids_dims.size() == 2);
    ORT_ENFORCE(scores_dims.size() == 2);
    int64_t batch_size = input_ids_dims[0];
    int64_t cur_len = input_ids_dims[1];
    ORT_ENFORCE(scores_dims[0] == batch_size);
    int64_t vocab_size = scores_dims[1];

    if (cur_len + 1 < ngram_size_ || ngram_size_ <= 0) {
      return Status::OK();
    }

    const auto* input_ids_data = static_cast<const int64_t*>(input_ids->DataRaw(input_ids->DataType()));

    auto lambda = [&](int64_t b) {
      for (int64_t i = 0; i < cur_len; ++i) {
        if (i + ngram_size_ > cur_len) {
          break;
        }

        bool is_banned = true;
        for (int64_t j = 0; j < ngram_size_ - 1; ++j) {
          auto token_at_tail = input_ids_data[b * cur_len + i + j];
          auto token_to_cmp = input_ids_data[b * cur_len + cur_len + 1 - ngram_size_ + j];
          if (token_at_tail != token_to_cmp) {
            is_banned = false;
            break;
          }
        }

        if (is_banned) {
          auto token_id = static_cast<int64_t>(input_ids_data[b * cur_len + i + ngram_size_ - 1]);
          ORT_ENFORCE(token_id < vocab_size);
          scores_target[b * vocab_size + token_id] = -std::numeric_limits<float>::infinity();
        }
      }
    };

    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    concurrency::ThreadPool::TryParallelFor(
      tp, batch_size, static_cast<double>(cur_len * ngram_size_),
      [&lambda](ptrdiff_t first, ptrdiff_t last) {
        for (auto b = static_cast<int64_t>(first), end = static_cast<int64_t>(last); b < end; ++b) {
          lambda(b);
        }
      }
    );

    return Status::OK();
  }
 private:
  int64_t ngram_size_;
};
}  // namespace contrib
}  // namespace onnxruntime
