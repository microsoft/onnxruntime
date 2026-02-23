// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/utils.h"

#ifndef SHARED_PROVIDER
#include "core/framework/config_options.h"
#else
#include "core/providers/shared_library/provider_wrappedtypes.h"
#endif

#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

void SetupMlasBackendKernelSelectorFromConfigOptions(MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* config,
                                                     const ConfigOptions& config_options) {
  if (config == nullptr) {
    return;
  }

  config->use_kleidiai = config_options.GetConfigEntry(kOrtSessionOptionsMlasDisableKleidiai) != "1";
}

}  // namespace onnxruntime
