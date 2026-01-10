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

#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/weight_only_quant_op.h"
#include <cuda_runtime_api.h>

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
          cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape,
          typename MainloopScheduleType, typename EpilogueScheduleType>
void sm90_generic_mixed_gemm_kernelLauncher(ActivationType const* A, WeightType const* B,
                                            ScaleZeroType const* weight_scales, ScaleZeroType const* weight_zero_points, BiasType const* biases,
                                            float const alpha, OutputType* C, int m, int n, int k, int const group_size,
                                            onnxruntime::llm::cutlass_extensions::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
                                            cudaStream_t stream, int* occupancy = nullptr);

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
