// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/quantization/moe_helper.h"

namespace onnxruntime {
namespace contrib {

enum class ActivationType {
  Relu = 0,
  Gelu = 1,
  Silu = 2,
  Identity = 3,
  SwiGLU = 4,
};

class MoEBaseCPU {
 protected:
  MoEBaseCPU(const OpKernelInfo& op_kernel_info) {
    ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_).IsOK());

    std::string activation_type_str;
    ORT_ENFORCE(op_kernel_info.GetAttr<std::string>("activation_type", &activation_type_str).IsOK());
    if (activation_type_str == "relu") {
      activation_type_ = ActivationType::Relu;
    } else if (activation_type_str == "gelu") {
      activation_type_ = ActivationType::Gelu;
    } else if (activation_type_str == "silu") {
      activation_type_ = ActivationType::Silu;
    } else if (activation_type_str == "identity") {
      activation_type_ = ActivationType::Identity;
    } else if (activation_type_str == "swiglu") {
      activation_type_ = ActivationType::SwiGLU;
    } else {
      ORT_THROW("Unsupported MoE activation type: ", activation_type_str);
    }

    normalize_routing_weights_ = op_kernel_info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 0) == 1;

    use_sparse_mixer_ = op_kernel_info.GetAttrOrDefault<int64_t>("use_sparse_mixer", 0) == 1;
    if (use_sparse_mixer_) {
      ORT_ENFORCE(k_ == 2, "Sparse mixer only supports k=2");
    }

    // Read activation parameters for SwiGLU
    activation_alpha_ = op_kernel_info.GetAttrOrDefault<float>("activation_alpha", 1.702f);  // Default 1.702f for SwiGLU
    activation_beta_ = op_kernel_info.GetAttrOrDefault<float>("activation_beta", 1.0f);      // Default 1.0f for SwiGLU
  }

  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  int64_t k_;
  ActivationType activation_type_;
  float activation_alpha_;
  float activation_beta_;
};

}  // namespace contrib
}  // namespace onnxruntime
