// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class ZeroGradient final : public RocmKernel {
 public:
  ZeroGradient(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename T_GRAD>
class InPlaceAccumulator final : public RocmKernel {
 public:
  InPlaceAccumulator(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Implementation can be found in hip file, optimizers_impl.cu
template <typename T, typename T_GRAD>
void InPlaceAccumulatorImpl(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count);

}  // namespace rocm
}  // namespace onnxruntime