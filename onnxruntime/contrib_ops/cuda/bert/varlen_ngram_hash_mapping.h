// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class VarlenNGramHashMapping final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit VarlenNGramHashMapping(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t max_ngram_size_;
  int64_t n_head_per_ngram_;
  T pad_id_;
  bool reset_on_eos_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
