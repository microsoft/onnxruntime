// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class NGramHashMapping final : public OpKernel {
 public:
  explicit NGramHashMapping(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  T HistoryId(const T* past_data, int64_t b, int64_t slot, int64_t state_length, T missing_history_value) const;

  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  T pad_id_;
  int64_t reset_on_eos_;
};

}  // namespace contrib
}  // namespace onnxruntime
