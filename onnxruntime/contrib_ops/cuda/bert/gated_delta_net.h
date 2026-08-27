// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class GatedDeltaNet final : public onnxruntime::cuda::CudaKernel {
 public:
  GatedDeltaNet(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  gated_delta_net::UpdateRule update_rule_;
  gated_delta_net::GateActivation gate_activation_;
  gated_delta_net::BetaActivation beta_activation_;
  gated_delta_net::Engine forced_engine_;
  float scale_;
  int chunk_size_;
  int state_update_capacity_;
  bool qk_l2_norm_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
