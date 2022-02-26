// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "qorder_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class QOrderedAttention final : public CudaKernel, public AttentionBase {
 public:
  QOrderedAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_weight_;
  int order_bias_;
  int order_output_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime