// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class EyeLike final : public OpKernel {
 public:
  EyeLike(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr("k", &k_).IsOK()) {
      k_ = 0;
    }

    has_dtype_ = info.GetAttr("dtype", &dtype_).IsOK();
  }
  
  Status Compute(OpKernelContext* context) const override;

 private:
  bool has_dtype_;
  int64_t dtype_;
  int64_t k_;
};

}  //namespace onnxruntime
