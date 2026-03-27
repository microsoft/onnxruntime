// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/framework/config_options.h"
#else
#include "core/providers/shared_library/provider_api.h"
#endif

#include <cstdlib>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

inline size_t ParseSizeTFromConfigOrDefault(const ConfigOptions& config_options,
                                            const char* option_name,
                                            size_t default_value) {
  const std::string value = config_options.GetConfigOrDefault(option_name, "");
  if (value.empty()) {
    return default_value;
  }

  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') {
    return default_value;
  }

  return static_cast<size_t>(parsed);
}

inline void SetupMlasBackendKernelSelectorFromConfigOptions(MLAS_BACKEND_KERNEL_SELECTOR_CONFIG& config,
                                                            const ConfigOptions& config_options) {
  config.use_kleidiai = config_options.GetConfigOrDefault(kOrtSessionOptionsMlasDisableKleidiAi, "0") != "1";
  config.enable_depthwise_with_multiplier_kernel =
    config_options.GetConfigOrDefault(kOrtSessionOptionsMlasEnableDepthwiseWithMultiplierKernel, "1") != "0";
  config.nchwc_conv_max_input_channel_batch = ParseSizeTFromConfigOrDefault(
      config_options, kOrtSessionOptionsMlasNchwcConvMaxInputChannelBatch, 0);
}

}  // namespace onnxruntime
