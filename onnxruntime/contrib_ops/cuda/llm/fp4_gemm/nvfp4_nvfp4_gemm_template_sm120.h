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

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/packed_stride.hpp"

#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"

#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {
using namespace cute;

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_, typename CGA_M_, typename CGA_N_,
          typename CGA_K_>
size_t genericFp4GemmKernelLauncherSm120(void*, void const*, void const*, void const*,
                                         void const*, float const*, int, int, int, int,
                                         tkc::CutlassGemmConfig, char*, size_t const, cudaStream_t,
                                         int*) {
  static_assert(always_false<T>, "Kernel should be explicitly instantiated.");
  return 0;
};

#ifdef PLACEHOLDER_KERNELS

#define INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(T, CTA_M_, CTA_N_, CTA_K_, CGA_M_, CGA_N_, CGA_K_)                                                                                                          \
  template <>                                                                                                                                                                                                  \
  size_t genericFp4GemmKernelLauncherSm120<T, cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>,                                                                                                         \
                                           cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>(void* D, void const* A, void const* B,                                                                     \
                                                                                                    void const* input_sf, void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count, \
                                                                                                    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,      \
                                                                                                    int* occupancy) {                                                                                          \
    ORT_THROW(                                                                                                                                                                                                 \
        "[LLM Error][FP4 gemm Runner] TensorRT-LLM is not compiled with support for this Architecture.");                                                                                                      \
  }

#else

#define INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER_SM120(T, CTA_M_, CTA_N_, CTA_K_, CGA_M_, CGA_N_, CGA_K_)                                                                                                          \
  struct DeviceGemmFp4GemmSm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##_##CGA_M_##_##CGA_N_##_##CGA_K_ {                                                                                                        \
    using ElementD = CudaToCutlassTypeAdapter<T>::type;                                                                                                                                                        \
    using CTAShape = cute::Shape<cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>>;                                                                                                                     \
    using ClusterShape = cute::Shape<_1, _1, _1>;                                                                                                                                                              \
    using Arch = cutlass::arch::Sm120;                                                                                                                                                                         \
    /* // Input A */                                                                                                                                                                                           \
    using ElementA = cutlass::float_e2m1_t;                                                                                                                                                                    \
    using LayoutA = cutlass::layout::RowMajor;                                                                                                                                                                 \
    static constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;                                                                                                                          \
    /* // Input B */                                                                                                                                                                                           \
    using ElementB = cutlass::float_e2m1_t;                                                                                                                                                                    \
    using LayoutB = cutlass::layout::ColumnMajor;                                                                                                                                                              \
    static constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;                                                                                                                          \
    /* // Input C */                                                                                                                                                                                           \
    using ElementC = void;                                                                                                                                                                                     \
    using LayoutC = cutlass::layout::ColumnMajor;                                                                                                                                                              \
    using LayoutD = cutlass::layout::RowMajor;                                                                                                                                                                 \
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;                                                                                                                             \
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;                                                                                                                             \
    using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;                                                                                                                                          \
    using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;                                                                                                                                          \
    using SFType = cutlass::float_ue4m3_t;                                                                                                                                                                     \
    using FusionOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>;                                                                                                        \
    using ElementCompute = float;                                                                                                                                                                              \
    using ElementAccumulator = float;                                                                                                                                                                          \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<Arch,                                                                                                                 \
                                                                                         cutlass::arch::OpClassTensorOp, CTAShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,              \
                                                                                         ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,                     \
                                                                                         cutlass::epilogue::TmaWarpSpecialized, FusionOperation>::CollectiveOp;                                                \
                                                                                                                                                                                                               \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<Arch,                                                                                                                     \
                                                                                     cutlass::arch::OpClassBlockScaledTensorOp, ElementPairA, LayoutA, AlignmentA, ElementPairB, LayoutB,                      \
                                                                                     AlignmentB, ElementAccumulator, CTAShape, ClusterShape,                                                                   \
                                                                                     cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(                                                       \
                                                                                         sizeof(typename CollectiveEpilogue::SharedStorage))>,                                                                 \
                                                                                     cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;                                                        \
    using TileSchedulerTag = cutlass::gemm::StaticPersistentScheduler;                                                                                                                                         \
    template <typename Base>                                                                                                                                                                                   \
    struct Sm12xOnly : Base {                                                                                                                                                                                  \
      using typename Base::Params;                                                                                                                                                                             \
      CUTLASS_DEVICE                                                                                                                                                                                           \
      void operator()(Params const& params, char* smem_buf) {                                                                                                                                                  \
        if constexpr (onnxruntime::llm::kernels::arch::is_major_v<12>) {                                                                                                                                       \
          this->Base::operator()(params, smem_buf);                                                                                                                                                            \
        } else {                                                                                                                                                                                               \
          if (cute::thread0()) {                                                                                                                                                                               \
            printf("%s : This kernel shall only run on SM12x devices.\n", __PRETTY_FUNCTION__);                                                                                                                \
            __trap();                                                                                                                                                                                          \
          }                                                                                                                                                                                                    \
        }                                                                                                                                                                                                      \
      }                                                                                                                                                                                                        \
    };                                                                                                                                                                                                         \
    using GemmKernel = Sm12xOnly<cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,                                                                                                         \
                                                                      CollectiveMainloop, CollectiveEpilogue, TileSchedulerTag>>;                                                                              \
                                                                                                                                                                                                               \
    using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                                                                                                                             \
  };                                                                                                                                                                                                           \
                                                                                                                                                                                                               \
  template <typename Gemm>                                                                                                                                                                                     \
  typename Gemm::Arguments                                                                                                                                                                                     \
  prepareGemmArgs_Sm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##_##CGA_M_##_##CGA_N_##_##CGA_K_(void* D,                                                                                                         \
                                                                                              void const* A, void const* B, void const* input_sf, void const* weight_sf, float const* global_sf, int m,        \
                                                                                              int n, int k, int batch_count) {                                                                                 \
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;                                                                                                          \
    using ElementA = typename Gemm::ElementA;                                                                                                                                                                  \
    using ElementB = typename Gemm::ElementB;                                                                                                                                                                  \
    using ElementSFA = cutlass::float_ue4m3_t;                                                                                                                                                                 \
    using ElementSFB = cutlass::float_ue4m3_t;                                                                                                                                                                 \
    using ElementC = void;                                                                                                                                                                                     \
    using ElementD = typename Gemm::ElementD;                                                                                                                                                                  \
    using ElementCompute = float;                                                                                                                                                                              \
                                                                                                                                                                                                               \
    typename Gemm::Arguments operator_args;                                                                                                                                                                    \
    operator_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;                                                                                                                                              \
    auto& fusion_args = operator_args.epilogue.thread;                                                                                                                                                         \
    fusion_args.alpha_ptr = static_cast<ElementCompute const*>(global_sf);                                                                                                                                     \
                                                                                                                                                                                                               \
    operator_args.problem_shape = cute::make_shape(m, n, k, batch_count);                                                                                                                                      \
                                                                                                                                                                                                               \
    operator_args.mainloop.ptr_A = static_cast<ElementA const*>(A);                                                                                                                                            \
    operator_args.mainloop.ptr_B = static_cast<ElementB const*>(B);                                                                                                                                            \
    operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const*>(input_sf);                                                                                                                                 \
    operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const*>(weight_sf);                                                                                                                                \
    operator_args.epilogue.ptr_C = static_cast<ElementC const*>(D);                                                                                                                                            \
    operator_args.epilogue.ptr_D = static_cast<ElementD*>(D);                                                                                                                                                  \
                                                                                                                                                                                                               \
    int const stride_A = batch_count == 1 ? 0 : m * k;                                                                                                                                                         \
    int const stride_B = batch_count == 1 ? 0 : n * k;                                                                                                                                                         \
    int const stride_C = batch_count == 1 ? 0 : m * n;                                                                                                                                                         \
                                                                                                                                                                                                               \
    operator_args.mainloop.dA = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, stride_A);                                                                                                    \
    operator_args.mainloop.dB = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, stride_B);                                                                                                    \
    operator_args.epilogue.dC = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideC>(n, stride_C);                                                                                                    \
    operator_args.epilogue.dD = operator_args.epilogue.dC;                                                                                                                                                     \
                                                                                                                                                                                                               \
    operator_args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);                                                                                             \
    operator_args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);                                                                                             \
                                                                                                                                                                                                               \
    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {                                                                                                                      \
      operator_args.scheduler.max_swizzle_size = 1;                                                                                                                                                            \
    }                                                                                                                                                                                                          \
    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {                                                                                                                          \
      using Enum_t = decltype(operator_args.scheduler.raster_order);                                                                                                                                           \
      operator_args.scheduler.raster_order = Enum_t::Heuristic;                                                                                                                                                \
    }                                                                                                                                                                                                          \
    operator_args.hw_info.cluster_shape = dim3(CGA_M_, CGA_N_, CGA_K_);                                                                                                                                        \
    operator_args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);                                                                                                                                              \
                                                                                                                                                                                                               \
    return operator_args;                                                                                                                                                                                      \
  }                                                                                                                                                                                                            \
                                                                                                                                                                                                               \
  template <>                                                                                                                                                                                                  \
  size_t genericFp4GemmKernelLauncherSm120<T, cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>,                                                                                                         \
                                           cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>(void* D, void const* A, void const* B,                                                                     \
                                                                                                    void const* input_sf, void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count, \
                                                                                                    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,      \
                                                                                                    int* occupancy) {                                                                                          \
    using ElementOutput__ = typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value,                                                                                                \
                                                                    cutlass::half_t, T>::type;                                                                                                                 \
    using ElementOutput_ =                                                                                                                                                                                     \
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput__, float>::value, float,                                                                                              \
                                                ElementOutput__>::type;                                                                                                                                        \
    using ElementOutput =                                                                                                                                                                                      \
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, SafeBF16>::value,                                                                                                   \
                                                cutlass::bfloat16_t, ElementOutput_>::type;                                                                                                                    \
                                                                                                                                                                                                               \
    using Fp4GemmOperator = DeviceGemmFp4GemmSm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##_##CGA_M_##_##CGA_N_##_##CGA_K_::Gemm;                                                                                \
    Fp4GemmOperator gemm;                                                                                                                                                                                      \
    auto args = prepareGemmArgs_Sm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##_##CGA_M_##_##CGA_N_##_##CGA_K_<                                                                                                   \
        Fp4GemmOperator>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count);                                                                                                                       \
    /* // Check shared memory size; throw when SMEM exceeds */                                                                                                                                                 \
    int smem_size = int(sizeof(typename Fp4GemmOperator::GemmKernel::SharedStorage));                                                                                                                          \
    static int mMaxSmemSize = tk::getMaxSharedMemoryPerBlockOptin();                                                                                                                                           \
    if (smem_size > mMaxSmemSize) {                                                                                                                                                                            \
      std::string errMsg = "SMEM size exceeds maximum allowed. Required " + std::to_string(smem_size) + ", got " + std::to_string(mMaxSmemSize);                                                               \
      ORT_THROW("[LLM Error][FP4 gemm Runner] " + errMsg);                                                                                                                                                     \
    }                                                                                                                                                                                                          \
    /* // Return workspace size */                                                                                                                                                                             \
    if (!A && !B && !D) {                                                                                                                                                                                      \
      return gemm.get_workspace_size(args);                                                                                                                                                                    \
    }                                                                                                                                                                                                          \
    if (gemm.get_workspace_size(args) > workspaceBytes) {                                                                                                                                                      \
      std::string errMsg("Requested workspace size insufficient. Required " + std::to_string(gemm.get_workspace_size(args)) + ", got " + std::to_string(workspaceBytes));                                      \
      ORT_THROW("[LLM Error][FP4 gemm Runner] " + errMsg);                                                                                                                                                     \
    }                                                                                                                                                                                                          \
    auto initStatus = gemm.initialize(args, workspace);                                                                                                                                                        \
    if (initStatus != cutlass::Status::kSuccess) {                                                                                                                                                             \
      auto cudaErrMsg = cudaGetErrorString(cudaGetLastError());                                                                                                                                                \
      std::string errMsg = "Failed to initialize cutlass FP4 gemm. Error: " + std::string(cutlass::cutlassGetStatusString(initStatus)) + " " + cudaErrMsg;                                                     \
      ORT_THROW("[LLM Error][FP4 gemm Runner] " + errMsg);                                                                                                                                                     \
    }                                                                                                                                                                                                          \
    auto runStatus = gemm.run(args, workspace, stream, nullptr, onnxruntime::llm::common::getEnvEnablePDL());                                                                                                  \
    if (runStatus != cutlass::Status::kSuccess) {                                                                                                                                                              \
      std::string errMsg = "Failed to run cutlass FP4 gemm. Error: " + std::string(cutlass::cutlassGetStatusString(runStatus));                                                                                \
      ORT_THROW("[LLM Error][FP4 gemm Runner] " + errMsg);                                                                                                                                                     \
    }                                                                                                                                                                                                          \
    return gemm.get_workspace_size(args);                                                                                                                                                                      \
  }

#endif

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
