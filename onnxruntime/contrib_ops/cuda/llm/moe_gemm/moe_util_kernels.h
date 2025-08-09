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
#include "contrib_ops/cuda/llm/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
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

// These kernels are used in moeUtilOp.cpp
int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node);

bool fusedBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* unpermuted_token_selected_experts,
                                        int* permuted_source_token_ids, int64_t* expert_first_token_offset, int64_t const num_tokens,
                                        int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
                                        cudaStream_t stream);

void threeStepBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* permuted_token_selected_experts,
                                            int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
                                            int* blocked_expert_counts, int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row,
                                            int64_t const num_tokens, int64_t const num_experts_per_node, int64_t const num_experts_per_token,
                                            int const start_expert_id, cudaStream_t stream);

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input,
                                   ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
                                   int const* permuted_row_to_unpermuted_row, int64_t const num_rows, int64_t const hidden_size, int const k,
                                   int const num_experts_per_node, QuantParams const& quant_params, bool use_per_expert_act_scale,
                                   int64_t* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
                                   TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, void const* prequant_scales, cudaStream_t stream);

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
                                      OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* final_scales,
                                      int const* unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
                                      int const* token_selected_experts, int64_t const* expert_first_token_offset, int64_t const num_rows,
                                      int64_t const cols, int64_t const experts_per_token, int64_t const num_experts_per_node,
                                      MOEParallelismConfig parallelism_config, bool const enable_alltoall, cudaStream_t stream);

}  // namespace cutlass_kernels
}  // namespace onnxruntime::llm::kernels
