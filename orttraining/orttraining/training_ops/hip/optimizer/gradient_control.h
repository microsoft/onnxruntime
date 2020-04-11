// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename T>
class ZeroGradient final : public HipKernel {
 public:
  ZeroGradient(const OpKernelInfo& info) : HipKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename T_GRAD>
class InPlaceAccumulator final : public HipKernel {
 public:
  InPlaceAccumulator(const OpKernelInfo& info) : HipKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Implementation can be found in hip file, optimizers_impl.cu
template <typename T, typename T_GRAD>
void InPlaceAccumulatorImpl(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count);

}  // namespace hip
}  // namespace onnxruntime