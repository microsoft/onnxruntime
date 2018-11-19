// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ImageScaler final : public CudaKernel {
 public:
  ImageScaler(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float scale_;
  std::vector<float> bias_;
  IAllocatorUniquePtr<float> b_data_;  // gpu copy of bias
};

}  // namespace cuda
}  // namespace onnxruntime
