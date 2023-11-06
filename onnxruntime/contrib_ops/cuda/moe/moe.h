// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/ft_moe/moe_kernel.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class MoEBlock final : public CudaKernel {
 public:
  explicit MoEBlock(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
    ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_).IsOK());

    std::string activation_type_str;
    ORT_ENFORCE(op_kernel_info.GetAttr<std::string>("activation_type", &activation_type_str).IsOK());
    if (activation_type_str == "relu") {
      activation_type_ = fastertransformer::ActivationType::Relu;
    } else if (activation_type_str == "gelu") {
      activation_type_ = fastertransformer::ActivationType::Gelu;
    } else if (activation_type_str == "silu") {
      activation_type_ = fastertransformer::ActivationType::Silu;
    } else if (activation_type_str == "identity") {
      activation_type_ = fastertransformer::ActivationType::Identity;
    } else {
      ORT_THROW("Unsupported MoE activation type: ", activation_type_str);
    }
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t k_;
  fastertransformer::ActivationType activation_type_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
