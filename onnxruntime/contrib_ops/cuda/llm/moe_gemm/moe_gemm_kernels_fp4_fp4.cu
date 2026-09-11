/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
// NVFP4: FP4 e2m1 activations + FP4 e2m1 weights (native block-scaled FP4xFP4 grouped GEMM).
// Routes through the Blackwell SM120 block-scaled tensor op path (nv_float4_t with ue4m3 SF,
// IsMXFPX=false) inside dispatchMoeGemmSelectBiasTmaWarpSpecialized. The BF16/FP16 activation is
// quantized to NVFP4 (block size 16, per-block E4M3 scales) inside the runner's
// expandInputRowsKernel. Requires ENABLE_FP4 (CUDA >= 12.8).
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
#endif
#endif
}  // namespace onnxruntime::llm::kernels::cutlass_kernels
