// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/crop.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Crop final : public CropBase, public CudaKernel {
 public:
  Crop(const OpKernelInfo& info) : CropBase(info), CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
