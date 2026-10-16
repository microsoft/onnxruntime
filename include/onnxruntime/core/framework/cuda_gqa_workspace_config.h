// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <limits>
#include <string_view>

#include "core/common/parse_string.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

// Config retrieval stays local to the host or provider; no framework facade crosses the DLL boundary.
inline Status ParseCudaGqaWorkspaceMaxTotalSequenceLength(
    std::string_view value, int64_t& max_total_sequence_length) {
  int64_t parsed = 0;
  if (value.empty() || !TryParseStringWithClassicLocale(value, parsed) || parsed < 0 ||
      parsed > std::numeric_limits<int32_t>::max()) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        kOrtSessionOptionsCudaGqaWorkspaceMaxTotalSequenceLength,
        " must be a decimal integer in [0, INT32_MAX], but got: '", value, "'");
  }

  max_total_sequence_length = parsed;
  return Status::OK();
}

}  // namespace onnxruntime
