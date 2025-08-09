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
#ifdef ENABLE_FP4
#include "contrib_ops/cuda/llm/fp4_gemm/fp4_gemm_template.h"

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 64, 128, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 128, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 4, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 1, 2, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 1, 4, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 2, 1, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 2, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 2, 4, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 4, 2, 1, _2SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 256, 256, 4, 4, 1, _2SM)

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(float, 128, 128, 128, 1, 1, 1)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(float, 128, 128, 256, 1, 1, 1)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(float, 256, 128, 128, 1, 1, 1)

template class CutlassFp4GemmRunner<float, FP4GemmType::W4A4_NVFP4_NVFP4>;
template class CutlassFp4GemmRunner<float, FP4GemmType::W4A8_MXFP4_MXFP8>;

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
#endif
