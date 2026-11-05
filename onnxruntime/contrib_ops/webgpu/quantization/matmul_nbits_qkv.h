// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class MatMulNBitsQkv : public WebGpuKernel {
 public:
  explicit MatMulNBitsQkv(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    Nq_ = info.GetAttr<int64_t>("Nq");
    Nkv_ = info.GetAttr<int64_t>("Nkv");
    block_size_ = info.GetAttr<int64_t>("block_size");
    bits_ = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    lora_bits_ = info.GetAttrOrDefault<int64_t>("lora_bits", 8);
    lora_block_size_ = info.GetAttrOrDefault<int64_t>("lora_block_size", 32);
    // GetInputCount is available through the plugin EP API. The schema checks
    // that the optional adapter inputs are either all present or all absent.
    has_adapter_inputs_ = info.GetInputCount() > 12;
    ORT_ENFORCE(!has_adapter_inputs_ || info.GetInputCount() == 24,
                "MatMulNBitsQkv requires all adapter inputs when any is present.");
    ORT_ENFORCE(bits_ == 4,
                "MatMulNBitsQkv currently supports 4-bit weights only.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  Status ComputeBase(onnxruntime::webgpu::ComputeContext& context) const;
  Status ComputeWithAdapters(onnxruntime::webgpu::ComputeContext& context) const;

  int64_t K_;
  int64_t Nq_;
  int64_t Nkv_;
  int64_t block_size_;
  int64_t accuracy_level_;
  int64_t bits_;
  float epsilon_;
  int64_t lora_bits_;
  int64_t lora_block_size_;
  bool has_adapter_inputs_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
