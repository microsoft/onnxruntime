// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

namespace onnxruntime {

/**
 * Gets the names of all execution providers, in order of decreasing default
 * priority.
 */
const std::vector<std::string>& GetAllExecutionProviderNames();

/**
 * Gets the names of execution providers available in this build, in order of
 * decreasing default priority.
 */
const std::vector<std::string>& GetAvailableExecutionProviderNames();

}  // namespace onnxruntime
