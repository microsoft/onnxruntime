// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/miopen_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T, typename T1, typename T2>
class BatchNormInternal final : public RocmKernel {
 public:
  BatchNormInternal(const OpKernelInfo& op_kernel_info)
      : RocmKernel{op_kernel_info},
        miopen_batch_norm_mode_(miopenBNSpatial),
        momentum_(0.9) {
    float tmp_epsilon;
    ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
    epsilon_ = ClampMiopenBatchNormEpsilon(static_cast<double>(tmp_epsilon));

    // spatial or not
    int64_t tmp_spatial;
    if (op_kernel_info.GetAttr<int64_t>("spatial", &tmp_spatial).IsOK()) {
      spatial_ = tmp_spatial;
    }

    if (spatial_ == 0) {
      miopen_batch_norm_mode_ = miopenBNPerActivation;
    }

    float tmp_momentum;
    if (op_kernel_info.GetAttr<float>("momentum", &tmp_momentum).IsOK()) {
      momentum_ = static_cast<double>(tmp_momentum);
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  double epsilon_;
  int64_t spatial_ = 1;  // default as per spec
  miopenBatchNormMode_t miopen_batch_norm_mode_;
  double momentum_;
};

}  // namespace rocm
}  // namespace onnxruntime
