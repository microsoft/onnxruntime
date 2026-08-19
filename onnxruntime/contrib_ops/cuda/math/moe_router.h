// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// The MoE routing decision, from the gate GEMM's output to the two tensors QMoE
// needs: the log-domain router row for this rank's expert columns, and the local weight mass
// that has to be multiplied back in afterwards.  Scoring and selection are attributes, so
// softmax/sigmoid routers and DeepSeek-style sqrt-softplus + noaux_tc both map onto it.
template <typename T>
class MoERouter final : public CudaKernel {
 public:
  explicit MoERouter(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t topk_;
  int64_t local_expert_start_;
  int64_t local_expert_count_;
  float route_scale_;
  // "noaux_tc" adds `bias` before the top-k; "topk" selects on the affinity alone.
  bool add_bias_before_topk_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
