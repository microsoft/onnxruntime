// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {

/**
 * When session.ep_cache_use_ort_version is "1", rewrites known EP cache path entries in
 * config_options to be suffixed with the ORT version (e.g. ".caches" -> ".caches/1.20.0").
 * Call this at the start of session initialization so that provider policy and any
 * late-added options use versioned paths.
 */
void ApplyEpCacheVersionToConfigOptions(ConfigOptions& config_options);

/**
 * Returns a copy of provider_options with known cache path values suffixed by the ORT version
 * when session.ep_cache_use_ort_version is "1" in config_options. Use this when creating an
 * execution provider factory so the factory receives versioned paths.
 * provider_name is the canonical EP name (e.g. "CoreML", "TensorRT").
 */
ProviderOptions GetProviderOptionsWithVersionedCachePaths(
    const ProviderOptions& provider_options,
    const ConfigOptions& config_options,
    const char* provider_name);

}  // namespace onnxruntime
