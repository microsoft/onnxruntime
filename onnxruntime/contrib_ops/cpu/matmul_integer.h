// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename T2, typename T3>
class MatMulInteger final : public OpKernel {
 public:
  MatMulInteger(const OpKernelInfo& info) : OpKernel(info) {
    if (info.GetInputCount() > 2) {
      has_a_zero_point_ = true;
      has_b_zero_point_ = true;
    } else {
      has_a_zero_point_ = false;
      has_b_zero_point_ = false;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool has_a_zero_point_;
  bool has_b_zero_point_;
};
}  // namespace contrib
}  // namespace onnxruntime