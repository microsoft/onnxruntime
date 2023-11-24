/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "ft_gemm_configs.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/common/common.h"

using namespace onnxruntime;

namespace ort_fastertransformer {

std::vector<CutlassGemmConfig> get_candidate_configs(int sm, const bool is_weight_only, const bool simt_configs_only);

CutlassGemmConfig estimate_best_config_from_occupancies(const std::vector<CutlassGemmConfig>& candidate_configs,
                                                        const std::vector<int>& occupancies, const int64_t m,
                                                        const int64_t n, const int64_t k, const int64_t num_experts,
                                                        const int split_k_limit, const size_t workspace_bytes,
                                                        const int multi_processor_count, const int is_weight_only);

}  // namespace ort_fastertransformer
