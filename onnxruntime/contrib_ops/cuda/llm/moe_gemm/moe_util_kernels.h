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
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "cutlass/gemm/gemm.h"
#include "core/common/common.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/quantization.h"
#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

#include "contrib_ops/cuda/llm/nv_infer_datatype.h"

#include <array>
#include <cuda_runtime_api.h>
#include <map>
#include <optional>
#include <random>
#include <utility>

namespace onnxruntime::llm::kernels {

namespace cutlass_kernels {

// Utility kernels used by the MoE runner to build expert maps, expand rows, and finalize routing.
int64_t computeNumTokensPerBlock(const int64_t num_tokens, const int64_t num_experts_per_node);

bool fusedBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_row_to_unpermuted_row,
                                        int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
                                        int64_t* expert_first_token_offset, const int64_t num_tokens,
                                        const int num_experts_per_node, const int experts_per_token, const int start_expert, const int end_expert,
                                        cudaStream_t stream);

void threeStepBuildExpertMapsSortFirstToken(const int* token_selected_experts, int* permuted_token_selected_experts,
                                            int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
                                            int* blocked_expert_counts, int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row,
                                            const int64_t num_tokens, const int64_t num_experts_per_node, const int64_t num_experts_per_token,
                                            const int start_expert_id, cudaStream_t stream);

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(const InputActivationsType* unpermuted_input,
                                   ExpandedActivationsType* permuted_output, const float* unpermuted_scales, float* permuted_scales,
                                   const int* permuted_row_to_unpermuted_row, const int64_t num_rows, const int64_t hidden_size, const int k,
                                   const int num_experts_per_node, const QuantParams& quant_params, bool use_per_expert_act_scale,
                                   int64_t* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
                                   const TmaWarpSpecializedGroupedGemmInput::ElementSF* input_sf, const void* prequant_scales, cudaStream_t stream);

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(const GemmOutputType* expanded_permuted_rows,
                                      OutputType* reduced_unpermuted_output, const ScaleBiasType* bias, const float* final_scales,
                                      const int* unpermuted_row_to_permuted_row, const int* permuted_row_to_unpermuted_row,
                                      const int* token_selected_experts, const int64_t* expert_first_token_offset, const int64_t num_rows,
                                      const int64_t cols, const int64_t experts_per_token, const int64_t num_experts_per_node,
                                      MOEParallelismConfig parallelism_config, const bool enable_alltoall, cudaStream_t stream);

}  // namespace cutlass_kernels
}  // namespace onnxruntime::llm::kernels
