// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

enum class MoEParallelType {
  None = 0,
  EP = 1,
  TP = 2,
  EPAndTP = 3,
};

enum class MoEQuantType {
  None = 0,
  UINT4 = 1,
  UINT8 = 2,
};

enum class ActivationType {
  Relu = 0,
  Gelu = 1,
  Silu = 2,
  Identity = 3,
  SwiGLU = 4,
};

struct MoEParameters {
  MoEParameters() {}
  explicit MoEParameters(int64_t tensor_shards) : tensor_shards(tensor_shards) {}
  int64_t num_rows;
  int64_t num_experts;
  int64_t local_num_experts;
  int64_t hidden_size;
  int64_t inter_size;

  MoEParallelType parallel_type;
  int64_t tensor_shards{1};
};

class MoEBaseCPU {
 public:
  Status CheckInputs(MoEParameters& parameters, MoEQuantType& quant_type, const Tensor* input,
                     const Tensor* router_probs, const Tensor* fc1_experts_weights,
                     const Tensor* fc1_experts_bias_optional, const Tensor* fc2_experts_weights,
                     const Tensor* fc2_experts_bias_optional, const Tensor* fc3_experts_weights_optional,
                     const Tensor* fc3_experts_bias_optional) const {
    ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);
    const auto& input_dims = input->Shape().GetDims();
    const auto& router_probs_dims = router_probs->Shape().GetDims();
    const auto& fc1_experts_weights_dims = fc1_experts_weights->Shape().GetDims();
    const auto& fc2_experts_weights_dims = fc2_experts_weights->Shape().GetDims();

    int64_t num_rows = input_dims.size() == 2 ? input_dims[0] : input_dims[0] * input_dims[1];
    int64_t hidden_size = input_dims[input_dims.size() - 1];
    int64_t local_num_experts = fc1_experts_weights_dims[0];
    int64_t num_experts = router_probs_dims[1];
    int64_t inter_size = fc2_experts_weights_dims[1];

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
                             fc2_experts_weights_dims[1], " and ", inter_size);
    }

    const int64_t coe = quant_type == MoEQuantType::UINT4 ? 2 : 1;
    const int64_t act = activation_type_ == ActivationType::SwiGLU ? 2 : 1;  // SwiGLU requires 2x weights for gate

    if (fc1_experts_weights_dims[2] != act * inter_size / coe) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "fc1_experts_weights_dims[2] is ", fc1_experts_weights_dims[2],
                             " expected ", act * inter_size / coe);
    }
    if (fc2_experts_weights_dims[2] != hidden_size / coe) {
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

    // Optional bias validation
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
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias_dims[0] must be equal to local_num_experts, got ",
                               fc1_experts_bias_dims[0], " and ", local_num_experts);
      }
      if (fc2_experts_bias_dims[0] != local_num_experts) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_bias_dims[0] must be equal to local_num_experts, got ",
                               fc2_experts_bias_dims[0], " and ", local_num_experts);
      }
      if (fc1_experts_bias_dims[1] != inter_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_bias_dims[1] must be equal to inter_size, got ",
                               fc1_experts_bias_dims[1], " and ", inter_size);
      }
      if (fc2_experts_bias_dims[1] != hidden_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_bias_dims[1] must be equal to hidden_size, got ",
                               fc2_experts_bias_dims[1], " and ", hidden_size);
      }
    }

    // FC3 validation - match CUDA FasterTransformer behavior
    if (activation_type_ == ActivationType::SwiGLU && fc3_experts_weights_optional != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "SwiGLU activation is not supported with fc3.");
    }
    if (fc3_experts_weights_optional != nullptr && activation_type_ != ActivationType::SwiGLU) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "FC3 gating is not yet implemented on CPU.");
    }

    // Set output parameters
    parameters.num_rows = num_rows;
    parameters.num_experts = num_experts;
    parameters.local_num_experts = local_num_experts;
    parameters.hidden_size = hidden_size;
    parameters.inter_size = inter_size;
    parameters.parallel_type = MoEParallelType::None;

    return Status::OK();
  }

  Status CheckInputScales(const Tensor* fc1_experts_scales, const Tensor* fc2_experts_scales, const Tensor* fc3_experts_scales_optional,
                          int64_t num_experts, int64_t hidden_size, int64_t inter_size) const {
    if (fc1_experts_scales == nullptr || fc2_experts_scales == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales and fc2_experts_scales cannot be null for quantized MoE");
    }

    // SwiGLU should not use separate FC3 scales - weights are concatenated in FC1
    if (activation_type_ == ActivationType::SwiGLU && fc3_experts_scales_optional != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "SwiGLU activation is not supported with fc3.");
    }
    if (activation_type_ != ActivationType::SwiGLU && fc3_experts_scales_optional != nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "FC3 gating is not yet implemented on CPU.");
    }

    const auto& fc1_experts_scales_dims = fc1_experts_scales->Shape().GetDims();
    const auto& fc2_experts_scales_dims = fc2_experts_scales->Shape().GetDims();

    if (fc1_experts_scales_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales must be 2D, got ",
                             fc1_experts_scales_dims.size());
    }
    if (fc2_experts_scales_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_scales must be 2D, got ",
                             fc2_experts_scales_dims.size());
    }
    if (fc1_experts_scales_dims[0] != num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales[0] must be equal to num_experts, got ",
                             fc1_experts_scales_dims[0], " and ", num_experts);
    }

    const int64_t act = activation_type_ == ActivationType::SwiGLU ? 2 : 1;  // SwiGLU requires 2x scales
    if (fc1_experts_scales_dims[1] != act * inter_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc1_experts_scales[1] is ", fc1_experts_scales_dims[1],
                             " expected ", act * inter_size);
    }
    if (fc2_experts_scales_dims[0] != num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_scales[0] must be equal to num_experts, got ",
                             fc2_experts_scales_dims[0], " and ", num_experts);
    }
    if (fc2_experts_scales_dims[1] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "fc2_experts_scales[1] must be equal to hidden_size, got ",
                             fc2_experts_scales_dims[1], " and ", hidden_size);
    }

    return Status::OK();
  }

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
  }

  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  int64_t k_;
  ActivationType activation_type_;
};

}  // namespace contrib
}  // namespace onnxruntime
