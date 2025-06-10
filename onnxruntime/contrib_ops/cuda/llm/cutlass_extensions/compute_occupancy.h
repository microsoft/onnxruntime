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

#include <cuda_runtime_api.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 177)
#endif

#include "cutlass/device_kernel.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::llm {
namespace cutlass_extensions {

template <typename GemmKernel, bool enable_cutlass_3x = false>
inline int compute_occupancy_for_kernel() {
  int smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));

  if (smem_size > (48 << 10)) {
    cudaFuncAttributes attr;
    int device = 0;
    int max_smem_per_block = 0;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    CUDA_CALL_THROW(
        cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if constexpr (enable_cutlass_3x) {
      CUDA_CALL_THROW(cudaFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>));
    } else {
      CUDA_CALL_THROW(cudaFuncGetAttributes(&attr, cutlass::Kernel<GemmKernel>));
    }
    if (smem_size + attr.sharedSizeBytes >= static_cast<size_t>(max_smem_per_block)) {
      // This should mean that
      // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
      // wouldn't work. In that case, we return an occupancy of 0. This will cause the heuristic to ignore this
      // configuration.
      return 0;
    }

    if constexpr (enable_cutlass_3x) {
      CUDA_CALL_THROW(cudaFuncSetAttribute(
          cutlass::device_kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    } else {
      CUDA_CALL_THROW(cudaFuncSetAttribute(
          cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
  }

  int max_active_blocks = -1;
  if constexpr (enable_cutlass_3x) {
    CUDA_CALL_THROW(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, cutlass::device_kernel<GemmKernel>,
                                                      128 * (GemmKernel::NumLoadWarpGroups + GemmKernel::NumMmaWarpGroups), smem_size));
  } else {
    CUDA_CALL_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, cutlass::Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size));
  }

  return max_active_blocks;
}

}  // namespace cutlass_extensions
}  // namespace onnxruntime::llm
