// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <unordered_map>
#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::cuda::CudaKernel;

class QuantizeWithOrder final : public CudaKernel {
 public:
  QuantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
};

class DequantizeWithOrder final : public CudaKernel {
 public:
  DequantizeWithOrder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int order_input_;
  int order_output_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime