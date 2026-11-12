// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string_view>

#include "core/common/common.h"

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

}  // namespace linear_attention_gates_helper
}  // namespace contrib
}  // namespace onnxruntime