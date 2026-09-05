// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class VarlenNGramHashMapping final : public OpKernel {
 public:
  explicit VarlenNGramHashMapping(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  T HistoryId(const T* past_data, int64_t b, int64_t slot, int64_t state_length) const;

  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  T pad_id_;
};

}  // namespace contrib
}  // namespace onnxruntime
