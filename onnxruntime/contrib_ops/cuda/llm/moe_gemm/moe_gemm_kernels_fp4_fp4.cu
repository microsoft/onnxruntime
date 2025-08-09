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

#if 0

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
#ifdef ENABLE_FP4
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
#endif
#endif

}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif