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

#include "contrib_ops/cuda/moe/cutlass_extensions/gemm_configs.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/common/common.h"

using namespace onnxruntime;

namespace ort_fastertransformer {

std::vector<CutlassGemmConfig> get_candidate_configs(int sm, bool const is_weight_only, bool const simt_configs_only,
                                                     bool const int8_configs_only = false, int const max_split_k = 1);

CutlassGemmConfig estimate_best_config_from_occupancies(std::vector<CutlassGemmConfig> const& candidate_configs,
                                                        std::vector<int> const& occupancies, const int64_t m,
                                                        const int64_t n, const int64_t k, const int64_t num_experts,
                                                        int const split_k_limit, const size_t workspace_bytes,
                                                        int const multi_processor_count, int const is_weight_only);

}  // namespace ort_fastertransformer
