// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels.h"
#include "contrib_ops/cpu/moe/moe_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class MoEBase {
 protected:
  MoEBase(const OpKernelInfo& op_kernel_info) {
    ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_).IsOK());

    std::string activation_type_str;
    ORT_ENFORCE(op_kernel_info.GetAttr<std::string>("activation_type", &activation_type_str).IsOK());
    if (activation_type_str == "relu") {
      activation_type_ = ort_fastertransformer::ActivationType::Relu;
    } else if (activation_type_str == "gelu") {
      activation_type_ = ort_fastertransformer::ActivationType::Gelu;
    } else if (activation_type_str == "silu") {
      activation_type_ = ort_fastertransformer::ActivationType::Silu;
    } else if (activation_type_str == "swiglu") {
      activation_type_ = ort_fastertransformer::ActivationType::SwiGLU;
    } else if (activation_type_str == "identity") {
      activation_type_ = ort_fastertransformer::ActivationType::Identity;
    } else {
      ORT_THROW("Unsupported MoE activation type: ", activation_type_str);
    }

    normalize_routing_weights_ = op_kernel_info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 0) == 1;

    use_sparse_mixer_ = op_kernel_info.GetAttrOrDefault<int64_t>("use_sparse_mixer", 0) == 1;
    if (use_sparse_mixer_) {
      ORT_ENFORCE(k_ == 2, "Sparse mixer only supports k=2");
    }

    // Parse SwiGLU-specific attributes. Defaults match the historical
    // hardcoded GPT-OSS values so that existing models continue to
    // produce the same output.
    //   swiglu_fusion = 1 : interleaved (gate, up, gate, up, ...)
    //   swiglu_fusion = 2 : chunked     (gate..., up...)
    //   activation_alpha  : multiplier applied to the gate before sigmoid
    //   activation_beta   : bias added to the up branch (linear + beta)
    //   swiglu_limit      : clamp range for gate (max) and up (±limit);
    //                       a non-positive value disables clamping.
    swiglu_fusion_ = op_kernel_info.GetAttrOrDefault<int64_t>("swiglu_fusion", 1);
    activation_alpha_ = op_kernel_info.GetAttrOrDefault<float>("activation_alpha", 1.702f);
    activation_beta_ = op_kernel_info.GetAttrOrDefault<float>("activation_beta", 1.0f);
    swiglu_limit_ = op_kernel_info.GetAttrOrDefault<float>("swiglu_limit", 7.0f);
    if (activation_type_ == ort_fastertransformer::ActivationType::SwiGLU) {
      ORT_ENFORCE(swiglu_fusion_ == 1 || swiglu_fusion_ == 2,
                  "swiglu_fusion must be 1 (interleaved) or 2 (chunked); got ", swiglu_fusion_);
    }
  }

  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  int64_t k_;
  ort_fastertransformer::ActivationType activation_type_;
  int64_t swiglu_fusion_;
  float activation_alpha_;
  float activation_beta_;
  float swiglu_limit_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
