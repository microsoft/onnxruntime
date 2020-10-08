// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
void SGDOptimizerImpl(
    const T* eta,
    const T* weights,
    const T* gradients,
    T* weight_out,
    T* gradients_out,
    size_t count);

class SGDOptimizer final : public RocmKernel {
 public:
  SGDOptimizer(const OpKernelInfo& info) : RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
