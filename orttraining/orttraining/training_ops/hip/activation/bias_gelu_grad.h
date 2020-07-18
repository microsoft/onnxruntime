// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename GeluComputationMode>
class BiasGeluGrad_dX : public HipKernel {
 public:
  BiasGeluGrad_dX(const OpKernelInfo& info) : HipKernel(info) {}
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

}  // namespace hip
}  // namespace onnxruntime
