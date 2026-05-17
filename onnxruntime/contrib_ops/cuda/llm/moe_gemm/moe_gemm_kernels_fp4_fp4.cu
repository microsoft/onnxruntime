/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
#if defined(ENABLE_FP4)
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
#endif
#endif
}  // namespace onnxruntime::llm::kernels::cutlass_kernels
