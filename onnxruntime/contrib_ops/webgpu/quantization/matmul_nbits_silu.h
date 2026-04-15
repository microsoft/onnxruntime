// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class MatMulNBitsSiluMul final : public WebGpuKernel {
 public:
  explicit MatMulNBitsSiluMul(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    block_size_ = info.GetAttr<int64_t>("block_size");
    bits_ = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    ORT_ENFORCE(bits_ == 4 || bits_ == 8 || bits_ == 2,
                "Only 4b/8b/2b quantization is supported for MatMulNBitsSiluMul op.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t accuracy_level_;
  int64_t bits_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime