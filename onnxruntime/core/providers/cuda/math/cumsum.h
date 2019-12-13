// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class CumSum final : public CudaKernel {
 public:
  explicit CumSum(const OpKernelInfo& info) : CudaKernel(info) {
    // Process exclusive attribute
    int64_t exclusive = 0;
    auto status = info.GetAttr("exclusive", &exclusive);
    if (status.IsOK()) {
      if (exclusive == 1 || exclusive == 0) {
        exclusive_ = (exclusive == 1);
      } else {
        ORT_ENFORCE("attribute exclusive can only be 0 or 1");
      }
    }

    // Process reverse attribute
    int64_t reverse = 0;
    status = info.GetAttr("reverse", &reverse);
    if (status.IsOK()) {
      if (reverse == 1 || reverse == 0) {
        reverse_ = (reverse == 1);
      } else {
        ORT_ENFORCE("attribute reverse can only be 0 or 1");
      }
    }
  }

  ~CumSum() = default;

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool exclusive_ = false;
  bool reverse_ = false;
};

}  // namespace cuda
}  // namespace onnxruntime
