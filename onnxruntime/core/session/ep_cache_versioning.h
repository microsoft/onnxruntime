// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <initializer_list>
#include <string>
#include <vector>

#include "core/framework/config_options.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {

using EpCachePathOptionKeys = std::vector<std::string>;

/**
 * Register the set of provider option keys that represent cache directory paths for a given EP.
 *
 * The provider_name must be the canonical EP name (e.g. "CoreML", "MIGraphX", "NvTensorRtRtx", "TensorRT").
 * Keys are the option names as they appear in ProviderOptions/config entries (e.g. "ModelCacheDirectory").
 *
 * This is typically called from EP-specific code (e.g. factory/bridge) so that EP-specific knowledge
 * about cache paths lives with the EP rather than in generic session code.
 */
void RegisterEpCachePathOptions(const char* provider_name,
                                std::initializer_list<const char*> option_keys);

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
