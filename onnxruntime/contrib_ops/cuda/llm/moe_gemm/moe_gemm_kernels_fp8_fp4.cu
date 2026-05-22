/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
// W4A8 / WFP4AFP8: FP8 e4m3 activations + MXFP4 weights.
// Routes through the SM100+ block-scaled tensor op path
// (OpClassBlockScaledTensorOp) inside dispatchMoeGemmSelectBiasTmaWarpSpecialized.
// Requires both ENABLE_FP4 (CUDA >= 12.8) and ENABLE_FP8.
#if defined(ENABLE_FP8) && defined(ENABLE_FP4) && defined(ENABLE_CUDA_FP4_QMOE) && defined(ENABLE_CUDA_FP8_QMOE)
template class MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16>;
#endif
#endif
}  // namespace onnxruntime::llm::kernels::cutlass_kernels
