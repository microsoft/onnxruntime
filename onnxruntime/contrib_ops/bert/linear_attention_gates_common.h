// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string_view>

#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {

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

}  // namespace contrib
}  // namespace onnxruntime
