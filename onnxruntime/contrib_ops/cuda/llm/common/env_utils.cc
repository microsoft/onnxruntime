/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>

namespace onnxruntime::llm::common {

std::optional<int32_t> getIntEnv(char const* name) {
  char const* const env = std::getenv(name);
  if (env == nullptr) {
    return std::nullopt;
  }
  int32_t const val = std::stoi(env);
  return {val};
};

std::optional<size_t> getUInt64Env(char const* name) {
  char const* const env = std::getenv(name);
  if (env == nullptr) {
    return std::nullopt;
  }
  size_t const val = std::stoull(env);
  return {val};
};

std::optional<std::string> getStrEnv(char const* name) {
  char const* const env = std::getenv(name);
  if (env == nullptr) {
    return std::nullopt;
  }
  return std::string(env);
}

// Returns true if the env variable exists and is set to "1"
bool getBoolEnv(char const* name) {
  char const* env = std::getenv(name);
  return env && env[0] == '1' && env[1] == '\0';
}

static std::string trim(std::string const& str) {
  size_t start = str.find_first_not_of(" \t\n\r");
  size_t end = str.find_last_not_of(" \t\n\r");
  return (start == std::string::npos || end == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

bool getEnvEnablePDL() {
  static std::once_flag flag;
  static bool enablePDL = false;

  std::call_once(flag,
                 [&]() {
                   if (getSMVersion() >= 90) {
                     // PDL will be enabled by setting the env variables `ORT_LLM_ENABLE_PDL` to `1`
                     enablePDL = getBoolEnv("ORT_LLM_ENABLE_PDL");
                   }
                 });
  return enablePDL;
}

}  // namespace onnxruntime::llm::common
