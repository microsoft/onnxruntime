// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

template <typename GeluComputationMode>
class BiasGeluGrad_dX : public RocmKernel {
 public:
  BiasGeluGrad_dX(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct KernelLaunchDispatcher {
    void operator()(
        int64_t input_size, int64_t bias_size,
        const Tensor& dY, const Tensor& X, const Tensor& B,
        Tensor& dX) const;
  };
};

}  // namespace rocm
}  // namespace onnxruntime
