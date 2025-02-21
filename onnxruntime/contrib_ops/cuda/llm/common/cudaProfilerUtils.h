/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

namespace onnxruntime::llm::common {

/// @brief Populate the start and end profiling iteration indexes from the provided environment variables
/// Try to set from envVarName first, and if that fails, try to set from legacyEnvVarName
/// Env variable values are expected to be in the format "1,2,3-5,6-8,9"
std::pair<std::unordered_set<int32_t>, std::unordered_set<int32_t>> populateIterationIndexes(
    std::string const& envVarName, std::optional<std::string> const& legacyEnvVarName = std::nullopt);

}  // namespace onnxruntime::llm::common
