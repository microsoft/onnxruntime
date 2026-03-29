// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class Attention final : public WebGpuKernel {
 public:
  Attention(const OpKernelInfo& info) : WebGpuKernel(info) {
    is_causal_ = info.GetAttrOrDefault<int64_t>("is_causal", 0);
    q_num_heads_ = info.GetAttrOrDefault<int64_t>("q_num_heads", 0);
    kv_num_heads_ = info.GetAttrOrDefault<int64_t>("kv_num_heads", 0);
    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
    softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
    qk_matmul_output_mode_ = info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0);
    softmax_precision_ = info.GetAttrOrDefault<int64_t>("softmax_precision", 0);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t is_causal_;
  int64_t q_num_heads_;
  int64_t kv_num_heads_;
  float scale_;
  float softcap_;
  int64_t qk_matmul_output_mode_;
  int64_t softmax_precision_;
};

}  // namespace webgpu
}  // namespace onnxruntime
