// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Upsample : public UpsampleBase, public CudaKernel {
 public:
  Upsample(const OpKernelInfo& info) : UpsampleBase(info), CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status BaseCompute(OpKernelContext* context, const std::vector<float>& roi, const std::vector<float>& scales,
                     const std::vector<int64_t>& output_dims) const;
};

}  // namespace cuda
}  // namespace onnxruntime
