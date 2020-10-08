// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class TransposeScaleMatMul final : public OpKernel {
 public:
  TransposeScaleMatMul(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_attr_;
  int64_t trans_a_attr_, trans_b_attr_;
};

}  // namespace contrib
}  // namespace onnxruntime
