// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class ClipGradNormInplace final : public CudaKernel {
 public:
  ClipGradNormInplace(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("max_norm", &max_norm_, 1.0f);
    info.GetAttrOrDefault("norm_type", &norm_type_, std::string("fro"));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float max_norm_;
  std::string norm_type_;
};

}  // namespace cuda
}  // namespace onnxruntime
