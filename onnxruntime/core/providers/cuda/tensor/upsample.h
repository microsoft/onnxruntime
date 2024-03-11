// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/upsamplebase.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Upsample : public UpsampleBase, public CudaKernel {
 public:
  explicit Upsample(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
  Status BaseCompute(OpKernelContext* context, gsl::span<const float> roi, gsl::span<const float> scales,
                     gsl::span<const int64_t> output_dims) const;

 private:
  IAllocatorUniquePtr<uint8_t> shared_lookup_table_ondevice_;
};

}  // namespace cuda
}  // namespace onnxruntime
