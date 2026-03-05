// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class CausalConv1DWithState final : public CudaKernel {
 public:
  CausalConv1DWithState(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string activation_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
