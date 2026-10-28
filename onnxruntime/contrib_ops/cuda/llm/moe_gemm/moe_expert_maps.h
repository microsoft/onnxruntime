/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#include <cuda_runtime_api.h>

namespace onnxruntime::llm::kernels::cutlass_kernels {

// Expert-map callers do not require the GEMM runner's types or CUTLASS headers.
int64_t computeNumTokensPerBlock(const int64_t num_tokens, const int64_t num_experts_per_node);

bool fusedBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                        int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                        int64_t* expert_first_token_offset, const int64_t num_tokens,
                                        const int num_experts_per_node, const int experts_per_token,
                                        const int start_expert, const int end_expert, cudaStream_t stream);

void threeStepBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_token_selected_experts,
                                            int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
                                            int64_t* expert_first_token_offset, int* blocked_expert_counts,
                                            int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row,
                                            const int64_t num_tokens, const int64_t num_experts_per_node,
                                            const int64_t num_experts_per_token, const int start_expert_id,
                                            cudaStream_t stream);

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
