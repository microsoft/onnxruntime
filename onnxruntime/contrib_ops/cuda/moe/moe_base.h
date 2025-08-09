// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/quantization/moe_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class MoEBase {
 protected:
  MoEBase(const OpKernelInfo& op_kernel_info, const cudaDeviceProp& device_prop) {
    ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_).IsOK());

    using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;
    std::string activation_type_str;
    ORT_ENFORCE(op_kernel_info.GetAttr<std::string>("activation_type", &activation_type_str).IsOK());
    if (activation_type_str == "relu") {
      activation_type_ = ActivationType::Relu;
    } else if (activation_type_str == "gelu") {
      activation_type_ = ActivationType::Gelu;
    } else if (activation_type_str == "silu") {
      activation_type_ = ActivationType::Silu;
    } else if (activation_type_str == "swiglu") {
      activation_type_ = ActivationType::SwiGLU;
    } else if (activation_type_str == "identity") {
      activation_type_ = ActivationType::Identity;
    } else {
      ORT_THROW("Unsupported MoE activation type: ", activation_type_str);
    }

    normalize_routing_weights_ = op_kernel_info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 0) == 1;

    use_sparse_mixer_ = op_kernel_info.GetAttrOrDefault<int64_t>("use_sparse_mixer", 0) == 1;
    if (use_sparse_mixer_) {
      ORT_ENFORCE(k_ == 2, "Sparse mixer only supports k=2");
    }

    sm_ = device_prop.major * 10 + device_prop.minor;
  }

  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  int64_t k_;
  onnxruntime::llm::kernels::cutlass_kernels::ActivationType activation_type_;
  int sm_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
