/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Auto-generated-style SM120 TMA Warp Specialized Grouped GEMM instantiations for FP8 activations with FP4 weights.
 */

#ifndef EXCLUDE_SM_120
#ifdef COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
#if defined(ENABLE_FP8) && defined(ENABLE_FP4) && defined(USE_FP4_QMOE)

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP8, SafeFP4, half, EpilogueOpDefault, NONE, 128, 128, 128, 1, 1, 1, true, false)

#ifdef ENABLE_BF16
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP8, SafeFP4, SafeBF16, EpilogueOpDefault, NONE, 128, 128, 128, 1, 1, 1, true, false)
#endif

}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif  // ENABLE_FP8 && ENABLE_FP4 && USE_FP4_QMOE
#endif  // COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
#endif  // EXCLUDE_SM_120
