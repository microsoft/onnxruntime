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

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/fused_moe_kernel.cuh"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

namespace onnxruntime::llm::kernels::cutlass_kernels
{
template <typename ElementType_, typename CutlassWeightType_, int MaxTileM_, int TileN_, int TileK_, int Stages_,
    typename EpilogueTag>
void sm80_generic_fused_moe_gemm_kernelLauncher(ElementType_ const* A, CutlassWeightType_ const* B,
    ElementType_ const* biases, bool bias_is_broadcast, ElementType_* C, int64_t const* total_tokens_including_expert,
    int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream,
    int* kernel_occupancy)
{
    constexpr auto activation_type = fused_moe::EpilogueRouting<EpilogueTag>(true);
    using GemmType = fused_moe::Fused_Moe_Kernel_sm80<ElementType_, CutlassWeightType_, ElementType_, MaxTileM_, TileN_,
        TileK_, Stages_, activation_type>;

    // make sure GPU has enough resources..
    if (kernel_occupancy != nullptr)
    {
        constexpr int smem_size = GemmType::kSmemSize;

        if (smem_size > (48 << 10))
        {
            cudaFuncAttributes attr{};
            int device = 0;
            int max_smem_per_block = 0;
            CUDA_CALL_THROW(cudaGetDevice(&device));
            CUDA_CALL_THROW(
                cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            CUDA_CALL_THROW(cudaFuncGetAttributes(&attr, fused_moe::run_global<GemmType>));
            if (smem_size + attr.sharedSizeBytes >= static_cast<size_t>(max_smem_per_block))
            {
                // This should mean that
                // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                // smem_size) wouldn't work. In that case, we return an occupancy of 0. This will cause the
                // heuristic to ignore this configuration.
                *kernel_occupancy = 0;
                return;
            }
        }

        int max_active_blocks = -1;
        CUDA_CALL_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, fused_moe::run_global<GemmType>, GemmType::kThreadCount, smem_size));
        *kernel_occupancy = max_active_blocks;
        return;
    }
    int occupancy = std::min(2, fused_moe::fused_gemm_maximum_active_blocks<GemmType>());
    int const threadblock_count = multi_processor_count * occupancy;
    ORT_ENFORCE(occupancy > 0, "GPU lacks the shared memory resources to run fused_moe kernel");
    using Arguments = typename GemmType::Arguments;
    Arguments args{{const_cast<ElementType_*>(A), const_cast<CutlassWeightType_*>(B), const_cast<ElementType_*>(biases),
                       reinterpret_cast<ElementType_*>(C), total_tokens_including_expert, static_cast<int>(gemm_n),
                       static_cast<int>(gemm_k), num_experts, bias_is_broadcast},
        num_experts, threadblock_count};
    auto params = GemmType::to_underlying_arguments(args);
    if (GemmType::kSmemSize >= (48 << 10))
    {
        cudaError_t result = cudaFuncSetAttribute(
            fused_moe::run_global<GemmType>, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmType::kSmemSize);
        ORT_ENFORCE(result == cudaSuccess,
            "Fail to set the max smem size to " + std::to_string(GemmType::kSmemSize) + " for fused moe kernel");
    }
    dim3 grid(params.threadblock_count, 1, 1);
    dim3 block(GemmType::kThreadCount);
    fused_moe::run_global<GemmType><<<grid, block, GemmType::kSmemSize, stream>>>(params);
    auto result = cudaGetLastError();
    ORT_ENFORCE(result == cudaSuccess, "Fail to execute fused moe kernel, cuda error %d\n", (int) (result));
}
} // namespace onnxruntime::llm::kernels::cutlass_kernels
