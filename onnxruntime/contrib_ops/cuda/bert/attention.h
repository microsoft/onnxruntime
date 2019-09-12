// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class Attention final : public CudaKernel {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int num_heads_;        // number of heads (N)
  int head_size_;        // size per head (H)
  int batch_size_;       // batch size (B)
  int sequence_length_;  // sequence length (S)

  IAllocatorUniquePtr<void> word_space_;  // gpu scratch buffer
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
