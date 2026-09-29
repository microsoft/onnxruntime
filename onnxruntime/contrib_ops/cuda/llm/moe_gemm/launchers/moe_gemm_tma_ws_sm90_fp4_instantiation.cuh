/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "contrib_ops/cuda/llm/common/logger.h"
#ifndef LLM_LOG_ERROR
#define LLM_LOG_ERROR(...) ORT_LLM_LOG_ERROR("mixed_input_launcher error")
#endif

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {

using EpiTag = onnxruntime::llm::cutlass_extensions::EpilogueOpDefault;
using EpiSched = cutlass::epilogue::TmaWarpSpecializedCooperative;
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using COOP = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
static constexpr auto QOP = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

// NONE fusion instantiation macros (Pingpong and Cooperative)
#define ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_PP(T, M, N, K, CM, CN, CK) \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<          \
      T, __nv_fp4_e2m1, T, EpiTag, EpilogueFusion::NONE,             \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,         \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>,      \
      PP, EpiSched, QOP>(                                            \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*)

#define ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_CO(T, M, N, K, CM, CN, CK) \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<          \
      T, __nv_fp4_e2m1, T, EpiTag, EpilogueFusion::NONE,             \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,         \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>,      \
      COOP, EpiSched, QOP>(                                          \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*)

// FINALIZE fusion instantiation macros (Pingpong and Cooperative)
#define ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_PP_FINALIZE(T, M, N, K, CM, CN, CK) \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<                   \
      T, __nv_fp4_e2m1, T, EpiTag, EpilogueFusion::FINALIZE,                  \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,                  \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>,               \
      PP, EpiSched, QOP>(                                                     \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*)

#define ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_CO_FINALIZE(T, M, N, K, CM, CN, CK) \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<                   \
      T, __nv_fp4_e2m1, T, EpiTag, EpilogueFusion::FINALIZE,                  \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,                  \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>,               \
      COOP, EpiSched, QOP>(                                                   \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*)

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
