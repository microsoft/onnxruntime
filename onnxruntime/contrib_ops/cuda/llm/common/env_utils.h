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

#pragma once

#include "core/platform/env_var_utils.h"

namespace onnxruntime::llm::common {
// Whether PDL is enabled.
static inline bool getEnvEnablePDL() {
  // PDL (Programmatic Dependent Launch) is only available on SM90+
  // Always return false for now as a safe default
  return false;
}

// Whether to force deterministic MOE.
static inline bool getEnvForceDeterministicMOE() {
  return ParseEnvironmentVariableWithDefault<int>("ORT_FORCE_DETERMINISTIC_MOE", 0) == 1;
}

}  // namespace onnxruntime::llm::common
