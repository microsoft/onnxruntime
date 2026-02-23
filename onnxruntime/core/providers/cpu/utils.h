// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/framework/config_options.h"
#else
#include "core/providers/shared_library/provider_api.h"
#endif

#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

inline void SetupMlasBackendKernelSelectorFromConfigOptions(MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* config,
                                                            const ConfigOptions& config_options) {
  if (config == nullptr) {
    return;
  }

  config->use_kleidiai = config_options.GetConfigEntry("mlas.disable_kleidiai") != "1";
}

}  // namespace onnxruntime
