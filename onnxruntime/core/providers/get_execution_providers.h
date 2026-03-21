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

/**
 * Checks whether a specific execution provider is usable at runtime.
 * Unlike GetAvailableExecutionProviderNames() which only reports compile-time
 * availability, this function checks that the provider's shared library file
 * exists on disk (for shared-library providers).
 *
 * This check does NOT load the library or initialize any hardware contexts.
 * It only verifies that the expected library file is present.
 *
 * In minimal builds, this falls back to compile-time availability (no
 * shared-library providers are used in minimal builds).
 *
 * @param provider_name The name of the execution provider to check
 *        (e.g. "CUDAExecutionProvider").
 * @return true if the provider is compiled in and its shared library file
 *         exists (or it is statically linked); false otherwise.
 */
bool IsExecutionProviderUsable(const std::string& provider_name);

/**
 * Gets the names of execution providers that are usable at runtime, in order
 * of decreasing default priority. For shared-library providers, this checks
 * that the library file exists on disk without loading it.
 *
 * In minimal builds, this falls back to compile-time availability.
 */
std::vector<std::string> GetUsableExecutionProviderNames();

}  // namespace onnxruntime
