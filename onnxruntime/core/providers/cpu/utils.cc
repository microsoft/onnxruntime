// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/utils.h"

#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

void SetUseKleidiaiFromConfigOptions(MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* config,
                                     const ConfigOptions& config_options) {
  if (config == nullptr) {
    return;
  }

  config->use_kleidiai = config_options.GetConfigEntry(kOrtSessionOptionsMlasDisableKleidiai) != "1";
}

}  // namespace onnxruntime
