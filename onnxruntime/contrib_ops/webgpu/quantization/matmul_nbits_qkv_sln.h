// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class MatMulNBitsQKVSimplifiedLayerNorm final : public WebGpuKernel {
 public:
  explicit MatMulNBitsQKVSimplifiedLayerNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    Nq_ = info.GetAttr<int64_t>("Nq");
    Nkv_ = info.GetAttr<int64_t>("Nkv");
    block_size_ = info.GetAttr<int64_t>("block_size");
    bits_ = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    ORT_ENFORCE(bits_ == 4,
                "MatMulNBitsQKVSimplifiedLayerNorm currently supports 4-bit weights only.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t Nq_;
  int64_t Nkv_;
  int64_t block_size_;
  int64_t accuracy_level_;
  int64_t bits_;
  float epsilon_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
