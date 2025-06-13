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

void preprocess_weights_for_mixed_gemm_cuda(cudaStream_t stream,
                                            int arch,
                                            int8_t* preprocessed_quantized_weight,
                                            int8_t* row_major_quantized_weight,
                                            int32_t* d_permutation_map,
                                            std::vector<size_t> const& shape,
                                            QuantType quant_type);

}  // namespace weight_only
}  // namespace kernels
}  // namespace onnxruntime::llm
