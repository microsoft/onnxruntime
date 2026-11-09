// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class GatedDeltaNet final : public OpKernel {
 public:
  explicit GatedDeltaNet(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  enum class UpdateRule : int {
    kLinear = 0,
    kGated = 1,
    kDelta = 2,
    kGatedDelta = 3,
  };
  enum class GateActivation : int {
    kNone = 0,
    kSoftplus = 1,
    kSigmoid = 2,
  };
  enum class BetaActivation : int {
    kNone = 0,
    kSigmoid = 1,
  };

  UpdateRule update_rule_;
  GateActivation gate_activation_;
  BetaActivation beta_activation_;
  float gate_lower_bound_;
  float scale_;
  int chunk_size_;
  int state_update_capacity_;
  bool qk_l2_norm_;
};

}  // namespace contrib
}  // namespace onnxruntime
