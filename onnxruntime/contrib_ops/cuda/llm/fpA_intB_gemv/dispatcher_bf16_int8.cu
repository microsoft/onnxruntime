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
#if USE_FPA_INTB_GEMM
#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true, 64);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
#endif
