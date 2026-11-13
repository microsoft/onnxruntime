// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <limits>
#include <string_view>

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace contrib {
namespace linear_attention_gates_helper {

enum class GatedRMSNormActivation : uint8_t {
  kSilu = 0,
  kSigmoid = 1,
};

inline Status ParseGatedRMSNormActivation(std::string_view activation, GatedRMSNormActivation& out) {
  if (activation == "silu" || activation == "swish") {
    out = GatedRMSNormActivation::kSilu;
    return Status::OK();
  }

  if (activation == "sigmoid") {
    out = GatedRMSNormActivation::kSigmoid;
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "activation must be one of: silu, swish, sigmoid");
}

inline GatedRMSNormActivation ParseGatedRMSNormActivationOrThrow(std::string_view activation) {
  GatedRMSNormActivation parsed_activation = GatedRMSNormActivation::kSilu;
  ORT_THROW_IF_ERROR(ParseGatedRMSNormActivation(activation, parsed_activation));
  return parsed_activation;
}

inline Status ValidateMoEGroupRouterAttributes(int64_t top_k, int64_t n_group,
                                               int64_t topk_group, int64_t group_topk) {
  ORT_RETURN_IF_NOT(top_k > 0, "top_k must be positive");
  ORT_RETURN_IF_NOT(n_group > 0, "n_group must be positive");
  ORT_RETURN_IF_NOT(topk_group > 0, "topk_group must be positive");
  ORT_RETURN_IF_NOT(group_topk > 0, "group_topk must be positive");
  return Status::OK();
}

inline Status ValidateMoEGroupRouterProjection(const TensorShape& input, const TensorShape& weight,
                                               const TensorShape& bias, int64_t top_k) {
  ORT_RETURN_IF_NOT(input.NumDimensions() == 2, "input must be rank 2, got ", input);
  ORT_RETURN_IF_NOT(weight.NumDimensions() == 2, "weight must be rank 2, got ", weight);
  const int64_t rows = input[0], width = input[1], experts = weight[0];
  ORT_RETURN_IF_NOT(rows >= 0 && rows <= std::numeric_limits<int>::max(),
                    "rows must be in [0, INT_MAX], got ", rows);
  ORT_RETURN_IF_NOT(width > 0 && width <= std::numeric_limits<int>::max() && width % 256 == 0,
                    "width must be a positive multiple of 256 and <= INT_MAX, got ", width);
  ORT_RETURN_IF_NOT(experts >= top_k && experts <= 512,
                    "k <= experts <= 512, got k=", top_k, " experts=", experts);
  ORT_RETURN_IF_NOT(weight[1] == width, "weight must be [experts, ", width, "], got ", weight);
  ORT_RETURN_IF_NOT(bias == TensorShape({experts}), "bias must be [", experts, "], got ", bias);
  return Status::OK();
}

}  // namespace linear_attention_gates_helper
}  // namespace contrib
}  // namespace onnxruntime