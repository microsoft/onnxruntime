// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

class ConfigOptions;

void SetupMlasBackendKernelSelectorFromConfigOptions(MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* config,
                                                     const ConfigOptions& config_options);

}  // namespace onnxruntime
