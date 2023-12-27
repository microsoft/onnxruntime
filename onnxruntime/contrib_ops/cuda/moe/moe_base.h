// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUTLASS

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cuda/moe/ft_moe/moe_gemm_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum class MoEParallelType {
  None = 0,
  ExpertSlicing = 1,
};

struct MoEParameters {
  int64_t num_rows;
  int64_t num_experts;
  int64_t local_num_experts;
  int64_t hidden_size;
  int64_t inter_size;
  MoEParallelType parallel_type;
};

class MoEBase {
 public:
  Status CheckInputs(MoEParameters& parameters,
                     const Tensor* input,
                     const Tensor* router_probs,
                     const Tensor* fc1_experts_weights,
                     const Tensor* fc2_experts_weights,
                     const Tensor* fc1_experts_bias_optional,
                     const Tensor* fc2_experts_bias_optional) const {
    const auto& input_dims = input->Shape().GetDims();
    const auto& router_probs_dims = router_probs->Shape().GetDims();
    const auto& fc1_experts_weights_dims = fc1_experts_weights->Shape().GetDims();
    const auto& fc2_experts_weights_dims = fc2_experts_weights->Shape().GetDims();

    int64_t num_rows = input_dims.size() == 2 ? input_dims[0] : input_dims[0] * input_dims[1];
    int64_t hidden_size = input_dims[input_dims.size() - 1];
    int64_t local_num_experts = fc1_experts_weights_dims[0];
    int64_t num_experts = router_probs_dims[1];
    int64_t inter_size = fc1_experts_weights_dims[2];

    if (fc1_experts_weights_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_weights_dims must be 3D, got ",
                             fc1_experts_weights_dims.size());
    }
    if (fc2_experts_weights_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_weights_dims must be 3D, got ",
                             fc2_experts_weights_dims.size());
    }
    if (fc1_experts_weights_dims[1] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc1_experts_weights_dims[1] must be equal to hidden_size, got ",
                             fc1_experts_weights_dims[1], " and ", hidden_size);
    }
    if (fc2_experts_weights_dims[1] != inter_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc2_experts_weights_dims[1] must be equal to inter_size, got ",
                             fc2_experts_weights_dims[1],
                             " and ", inter_size);
    }
    if (fc1_experts_weights_dims[2] != inter_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc1_experts_weights_dims[2] must be equal to inter_size, got ",
                             fc1_experts_weights_dims[2],
                             " and ", inter_size);
    }
    if (fc2_experts_weights_dims[2] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc2_experts_weights_dims[2] must be equal to hidden_size, got ",
                             fc2_experts_weights_dims[2], " and ", hidden_size);
    }
    if (router_probs_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "router_probs_dims must be 2D, got ",
                             router_probs_dims.size());
    }
    if (router_probs_dims[0] != num_rows) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "router_probs_dims[0] must be equal to num_rows, got ",
                             router_probs_dims[0], " and ", num_rows);
    }
    if (fc1_experts_bias_optional != nullptr && fc2_experts_bias_optional == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias is set but fc2_experts_bias is not set");
    }
    if (fc1_experts_bias_optional == nullptr && fc2_experts_bias_optional != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias is not set but fc2_experts_bias is set");
    }
    if (fc1_experts_bias_optional != nullptr && fc2_experts_bias_optional != nullptr) {
      const auto& fc1_experts_bias_dims = fc1_experts_bias_optional->Shape().GetDims();
      const auto& fc2_experts_bias_dims = fc2_experts_bias_optional->Shape().GetDims();
      if (fc1_experts_bias_dims.size() != 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias_dims must be 2D, got ",
                               fc1_experts_bias_dims.size());
      }
      if (fc2_experts_bias_dims.size() != 2) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_bias_dims must be 2D, got ",
                               fc2_experts_bias_dims.size());
      }
      if (fc1_experts_bias_dims[0] != local_num_experts) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "fc1_experts_bias_dims[0] must be equal to local_num_experts, got ",
                               fc1_experts_bias_dims[0],
                               " and ", local_num_experts);
      }
      if (fc2_experts_bias_dims[0] != num_experts) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "fc2_experts_bias_dims[0] must be equal to num_experts, got ",
                               fc2_experts_bias_dims[0],
                               " and ", num_experts);
      }
      if (fc1_experts_bias_dims[1] != inter_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "fc1_experts_bias_dims[1] must be equal to inter_size, got ",
                               fc1_experts_bias_dims[1],
                               " and ", inter_size);
      }
      if (fc2_experts_bias_dims[1] != hidden_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "fc2_experts_bias_dims[1] must be equal to hidden_size, got ",
                               fc2_experts_bias_dims[1],
                               " and ", hidden_size);
      }
    }

    parameters.num_rows = num_rows;
    parameters.num_experts = num_experts;
    parameters.local_num_experts = local_num_experts;
    parameters.hidden_size = hidden_size;
    parameters.inter_size = inter_size;
    if (num_experts == local_num_experts) {
      parameters.parallel_type = MoEParallelType::None;
    } else if (num_experts > local_num_experts) {
      parameters.parallel_type = MoEParallelType::ExpertSlicing;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "num_experts must be greater than or equal to local_num_experts, got ",
                             num_experts, " and ", local_num_experts);
    }

    return Status::OK();
  }

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

#endif
