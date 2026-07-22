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

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstddef>
#include <vector>

namespace onnxruntime::llm {
namespace kernels {
namespace weight_only {

enum class QuantType {
  W8_A16,
  W4_A16,
  W4_AFP8
};

int get_arch_for_mixed_gemm_weight_preprocess(int arch);

// ``apply_bias_interleave`` controls the final integer-only "+bias + pair-interleave" step
// (step 4). It MUST be left true for genuine signed-integer weights (INT4/INT8). For MXFP4
// (e2m1) codes the nibbles are floating-point, so the integer +8 bias would corrupt them;
// FP4 callers pass false to skip step 4 while keeping the layout-only steps 1-3 (row-permute,
// subbyte-transpose, column-interleave), which apply to e2m1 unchanged.
//
// ``interleave_without_bias`` (only consulted when ``apply_bias_interleave`` is false) applies
// step 4's [e0,e2,e4,e6,e1,e3,e5,e7] nibble pair-interleave WITHOUT the +8 bias. This is the
// layout the SM80 MoE grouped GEMM's e2m1 dequant converter expects (it inverts that
// permutation). The fused MXFP4 GEMV decode kernel uses a different layout and leaves this
// false.
void preprocess_weights_for_mixed_gemm_cuda(cudaStream_t stream,
                                            int arch,
                                            int8_t* preprocessed_quantized_weight,
                                            int8_t* row_major_quantized_weight,
                                            int32_t* d_permutation_map,
                                            std::vector<size_t> const& shape,
                                            QuantType quant_type,
                                            bool synchronize = true,
                                            bool apply_bias_interleave = true,
                                            bool interleave_without_bias = false);

}  // namespace weight_only
}  // namespace kernels
}  // namespace onnxruntime::llm
