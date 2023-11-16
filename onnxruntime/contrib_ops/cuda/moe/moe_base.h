// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct MoEParameters {
  int64_t num_rows;
  int64_t num_experts;
  int64_t hidden_size;
  int64_t inter_size;
};

class MoEBase {
 public:
  Status CheckInputs(MoEParameters& parameters,
                     const Tensor* input,
                     const Tensor* router_probs,
                     const Tensor* fc1_experts_weights,
                     const Tensor* fc2_experts_weights,
                     const Tensor* fc1_experts_bias_optional,
                     const Tensor* fc2_experts_bias_optional) const;

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
    } else if (activation_type_str == "identity") {
      activation_type_ = ort_fastertransformer::ActivationType::Identity;
    } else {
      ORT_THROW("Unsupported MoE activation type: ", activation_type_str);
    }
  }

  int64_t k_;
  ort_fastertransformer::ActivationType activation_type_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
