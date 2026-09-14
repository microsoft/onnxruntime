// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/moe/moe_base.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class MoE : public WebGpuKernel {
 public:
  MoE(const OpKernelInfo& info) : WebGpuKernel(info) {
    activation_alpha_ = static_cast<float>(info.GetAttrOrDefault<float>("activation_alpha", 1.0));
    activation_beta_ = static_cast<float>(info.GetAttrOrDefault<float>("activation_beta", 0.0));
    swiglu_fusion_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("swiglu_fusion", 0));
    swiglu_limit_ = info.GetAttrOrDefault<float>("swiglu_limit", std::numeric_limits<float>::infinity());
    k_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("k", 4));
    normalize_routing_weights_ = info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 0) == 1;
    use_sparse_mixer_ = info.GetAttrOrDefault<int64_t>("use_sparse_mixer", 0) == 1;
    std::string activation_type = info.GetAttrOrDefault<std::string>("activation_type", "relu");
    if (activation_type == "relu") {
      activation_type_ = MoEActivationType::Relu;
    } else if (activation_type == "gelu") {
      activation_type_ = MoEActivationType::Gelu;
    } else if (activation_type == "silu") {
      activation_type_ = MoEActivationType::Silu;
    } else if (activation_type == "identity") {
      activation_type_ = MoEActivationType::Identity;
    } else if (activation_type == "swiglu") {
      activation_type_ = MoEActivationType::SwiGLU;
    } else {
      ORT_THROW("Unsupported MoE activation type: ", activation_type);
    }

    ORT_ENFORCE(swiglu_fusion_ >= 0 && swiglu_fusion_ <= 2,
                "swiglu_fusion must be 0, 1, or 2, but got ", swiglu_fusion_);
    ORT_ENFORCE(activation_type_ == MoEActivationType::SwiGLU || swiglu_fusion_ == 0,
                "swiglu_fusion is only valid when activation_type is 'swiglu'.");
    ORT_ENFORCE(use_sparse_mixer_ == 0, "use_sparse_mixer not supported");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 protected:
  int k_;
  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  MoEActivationType activation_type_;
  int swiglu_fusion_;
  float swiglu_limit_;
  float activation_alpha_;
  float activation_beta_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
