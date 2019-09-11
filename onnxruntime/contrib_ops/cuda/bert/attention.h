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

class Attention final : public CudaKernel {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int numHeads_;        // number of heads (N)
  int headSize_;        // size per head (H)
  int batchSize_;       // batch size (B)
  int sequenceLength_;  // sequence length (S)

  IAllocatorUniquePtr<void> workSpace_;  // gpu scratch buffer
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
