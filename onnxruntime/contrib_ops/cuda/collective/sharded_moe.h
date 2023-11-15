// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/ft_moe/moe_kernel.h"
#include "core/common/common.h"
#include "nccl_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

using namespace onnxruntime::cuda;

template <typename T>
class ShardedMoE final : public NcclKernel {
 public:
  explicit ShardedMoE(const OpKernelInfo& op_kernel_info) : NcclKernel(op_kernel_info) {
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
      ORT_THROW("Unsupported ShardedMoE activation type: ", activation_type_str);
    }
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t k_;
  ort_fastertransformer::ActivationType activation_type_;
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
