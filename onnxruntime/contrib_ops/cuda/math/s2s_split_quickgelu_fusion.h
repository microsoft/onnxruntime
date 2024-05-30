// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Add Split + QuickGelu
class S2SModelSplitQuickGelu final : public CudaKernel {
 public:
  S2SModelSplitQuickGelu(const OpKernelInfo& info) : CudaKernel(info) {
   alpha = info.GetAttrOrDefault<float>("alpha", 1.702f);
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct KernelLaunchDispatcher {
   void operator()(cudaStream_t stream, int dim, int64_t input_size, float alpha, const Tensor& input, Tensor& output) const;
  };
  float alpha;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
