// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {
  
template <typename T>
class MatMul final : public HipKernel {
  using Base = HipKernel;

 public:
  MatMul(const OpKernelInfo& info)
      : HipKernel(info) {
    trans_A_ = info.GetAttrOrDefault<int64_t>("transA", 0);
    trans_B_ = info.GetAttrOrDefault<int64_t>("transB", 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool trans_A_;
  bool trans_B_;
};

}  // namespace hip
}  // namespace onnxruntime
