// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"
#include <limits>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

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
      activation_type_ = ActivationType::Swiglu;
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

    // Activation parameters for parameterized SwiGLU
    // Formula: G * sigmoid(alpha * G) * (L + beta)
    // Default alpha=1.0f gives standard silu: x * sigmoid(x)
    // Default beta=0.0f gives standard multiplication without offset
    activation_alpha_ = op_kernel_info.GetAttrOrDefault<float>("activation_alpha", 1.0f);
    activation_beta_ = op_kernel_info.GetAttrOrDefault<float>("activation_beta", 0.0f);

    // SwiGLU fusion mode: 0=not fused (fc1+fc3 separate), 1=fused interleaved, 2=fused chunked
    swiglu_fusion_ = static_cast<int>(op_kernel_info.GetAttrOrDefault<int64_t>("swiglu_fusion", 0));
    ORT_ENFORCE(swiglu_fusion_ >= 0 && swiglu_fusion_ <= 2,
                "swiglu_fusion must be 0, 1, or 2, but got ", swiglu_fusion_);
    ORT_ENFORCE(activation_type_ == ActivationType::Swiglu || swiglu_fusion_ == 0,
                "swiglu_fusion is only valid when activation_type is 'swiglu'.");

    // SwiGLU limit for clamping (optional, use infinity if not provided)
    swiglu_limit_ = op_kernel_info.GetAttrOrDefault<float>("swiglu_limit", std::numeric_limits<float>::infinity());

    block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", 0);

    sm_ = device_prop.major * 10 + device_prop.minor;
  }

  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  int64_t k_;
  onnxruntime::llm::kernels::cutlass_kernels::ActivationType activation_type_;
  float activation_alpha_;
  float activation_beta_;
  int swiglu_fusion_;   // 0: not fused, 1: fused interleaved, 2: fused chunked
  float swiglu_limit_;  // Clamp limit for SwiGLU
  int64_t block_size_;
  int sm_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
