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

#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {
template class CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t,
                                        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>;
}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
