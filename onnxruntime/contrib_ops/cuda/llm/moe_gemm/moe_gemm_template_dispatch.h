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

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cute/tensor.hpp"

#include <algorithm>

#include "cutlass/cutlass.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/threadblock/dq_mma_pipelined.h"

#include "cutlass/tensor_ref.h"

// #include "cutlass/gemm/device/gemm_grouped.h"
// #include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "contrib_ops/cuda/llm/cutlass_extensions/compute_occupancy.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/threadblock/default_mma.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/weight_only_quant_op.h"

#ifdef __GNUC__  // Restore GCC-specific diagnostics
#pragma GCC diagnostic pop
#endif

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"

#include "contrib_ops/cuda/llm/cutlass_heuristic.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.h"
#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_launcher.h"
#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch_tma_ws.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch_tma_ws_mixed_dtype.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_tma_warp_specialized_traits.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>
#include <type_traits>
#include <cstdio>

#define ORT_DEBUG_MOE 0

namespace onnxruntime::llm::kernels::cutlass_kernels {

#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
inline constexpr bool kCompileHopperTmaGroupedGemms = true;
#else
inline constexpr bool kCompileHopperTmaGroupedGemms = false;
#endif

#ifdef COMPILE_BLACKWELL_TMA_GROUPED_GEMMS
inline constexpr bool kCompileBlackwellTmaGroupedGemms = true;
#else
inline constexpr bool kCompileBlackwellTmaGroupedGemms = false;
#endif

#ifdef COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
inline constexpr bool kCompileBlackwellSm120TmaGroupedGemms = true;
#else
inline constexpr bool kCompileBlackwellSm120TmaGroupedGemms = false;
#endif

inline bool isTmaWarpSpecializedGroupedGemmCompiledForSm(int sm) {
  return (sm == 90 && kCompileHopperTmaGroupedGemms) ||
         (sm >= 100 && sm < 120 && kCompileBlackwellTmaGroupedGemms) ||
         ((sm == 120 || sm == 121) && kCompileBlackwellSm120TmaGroupedGemms);
}

// ============================= Variable batched Gemm things ===========================
template <typename T, typename WeightType, typename GemmOutputType, typename arch, cutlass::WeightOnlyQuantOp QuantOp,
          typename EpilogueTag, typename ThreadblockShape, typename WarpShape, int Stages>
struct genericMoeGemmKernelLauncher {
  static void call(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
    ORT_LLM_LOG_ENTRY();
#if defined(ENABLE_FP8)
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, __nv_fp8_e4m3>::value || cutlass::platform::is_same<T, __nv_fp8_e5m2>::value || cutlass::platform::is_same<T, float>::value,
                  "Specialized for fp8, bfloat16, half, float");
#elif defined(ENABLE_BF16)
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
                  "Specialized for bfloat16, half, float");
#else
    static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
                  "Specialized for half, float");
#endif

    static_assert(cutlass::platform::is_same<T, WeightType>::value || cutlass::platform::is_same<WeightType, uint8_t>::value || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value);

    static_assert(arch::kMinComputeCapability < 90, "Sm90+ architecture should use specialized kernels");

    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType = typename CudaToCutlassTypeAdapter<T>::type;
    using CutlassGemmOutputType = typename CudaToCutlassTypeAdapter<GemmOutputType>::type;
    using CutlassWeightType = typename CudaToCutlassTypeAdapter<WeightType>::type;
    if (!inputs.use_fused_moe) {
      // We need separate config for each architecture since we will target different tensorcore instructions. For
      // float, we do not target TCs.
      using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
      using ElementAccumulator = typename MixedGemmArchTraits::AccType;

      using EpilogueOp = typename onnxruntime::llm::cutlass_extensions::Epilogue<CutlassGemmOutputType,
                                                                                 MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

      typename EpilogueOp::Params epilogue_op(
          ElementAccumulator(1.f), inputs.biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));
      using TaggedOperator =
          typename cutlass::arch::TagOperator<typename MixedGemmArchTraits::Operator, QuantOp>::TaggedOperator;

#if defined(ENABLE_FP8)
      if constexpr ((std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) && std::is_same_v<EpilogueTag,
                                                                                                             cutlass_extensions::EpilogueOpDefault>) {
        if constexpr (std::is_same_v<T, WeightType>) {
          ORT_ENFORCE(inputs.scales == nullptr && inputs.biases == nullptr && inputs.alpha_scales,
                      "weight_scales and biases should be nullptr and alpha_scale_ptr_array shouldn't be nullptr for "
                      "FP8 "
                      "Ada.");
        } else {
          ORT_ENFORCE(
              inputs.alpha_scales, "alpha_scale_ptr_array shouldn't be nullptr for FP8 Ada.");
        }
        epilogue_op.alpha_ptr_array = inputs.alpha_scales;
      }
#endif

      // Finally, set up the kernel.
      using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementType,
                                                                             cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA,
                                                                             CutlassWeightType, typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
                                                                             MixedGemmArchTraits::ElementsPerAccessB, CutlassGemmOutputType, cutlass::layout::RowMajor,
                                                                             ElementAccumulator, typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
                                                                             typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
                                                                             cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
                                                                             cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, TaggedOperator>::GemmKernel;

      using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma,
                                                          typename GemmKernel_::Epilogue, typename GemmKernel_::ThreadblockSwizzle,
                                                          arch,  // Ensure top level arch is used for dispatch
                                                          GemmKernel_::kGroupScheduleMode>;

      using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

      if (inputs.occupancy != nullptr) {
        *inputs.occupancy = onnxruntime::llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
        return;
      }
      int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
      ORT_ENFORCE(occupancy > 0, "GPU lacks the shared memory resources to run GroupedGEMM kernel");
      int const threadblock_count = sm_count_ * occupancy;

      int const gemm_group_size = QuantOp == cutlass::WeightOnlyQuantOp::UNDEFINED ? inputs.k : inputs.groupwise_quant_group_size;
      typename GemmGrouped::Arguments args(inputs.num_experts, threadblock_count, gemm_group_size, epilogue_op,
                                           reinterpret_cast<ElementType const*>(inputs.A), reinterpret_cast<CutlassWeightType const*>(inputs.B),
                                           reinterpret_cast<CutlassGemmOutputType const*>(inputs.scales),
                                           reinterpret_cast<CutlassGemmOutputType const*>(inputs.zeros),
                                           reinterpret_cast<CutlassGemmOutputType const*>(inputs.biases), inputs.bias_is_broadcast,
                                           reinterpret_cast<CutlassGemmOutputType*>(inputs.C), inputs.total_tokens_including_expert, inputs.n,
                                           inputs.k);
#if ORT_DEBUG_MOE
      // Debug: Print GEMM dimensions and samples for float types
      if constexpr (std::is_same_v<T, float>) {
        // printf("DEBUG [GEMM float]: num_experts=%d, N=%lld, K=%lld, num_rows=%lld\n",
        //        inputs.num_experts, (long long)inputs.n, (long long)inputs.k, (long long)inputs.num_rows);

        // Copy offsets to host
        std::vector<int64_t> host_offsets(inputs.num_experts);
        // inputs.total_tokens_including_expert points to expert_first_token_offset + 1
        // correct pointer to start is inputs.total_tokens_including_expert - 1?
        // Wait, passing `expert_first_token_offset + 1` means [0] is offset of expert 1 start?
        // No, expert_first_token_offset[0] is 0.
        // expert_first_token_offset[1] is end of E0.
        // inputs.total_tokens_including_expert[0] is expert_first_token_offset[1] (end of E0).
        // inputs.total_tokens_including_expert[e] is end of E_e.
        // So start of E_e is (e==0) ? 0 : inputs.total_tokens_including_expert[e-1].

        cudaMemcpy(host_offsets.data(), inputs.total_tokens_including_expert, inputs.num_experts * sizeof(int64_t), cudaMemcpyDeviceToHost);

        for (int e = 0; e < inputs.num_experts; ++e) {
          int64_t start_row = (e == 0) ? 0 : host_offsets[e - 1];
          int64_t end_row = host_offsets[e];
          int64_t valid_rows = end_row - start_row;

          // printf("DEBUG [GEMM float] Expert %d: Valid Rows=%lld, Start Row=%lld\n", e, (long long)valid_rows, (long long)start_row);

          if (valid_rows > 0) {
            // Sample Input A (Row `start_row`)
            float a_sample = 0;
            // A is [TotalRows, K] (row major)
            // ptr = A + start_row * K
            const float* a_ptr = reinterpret_cast<const float*>(inputs.A) + start_row * inputs.k;
            cudaMemcpy(&a_sample, a_ptr, sizeof(float), cudaMemcpyDeviceToHost);

            // Sample Weight B for expert e
            // B is [Experts, N, K] or similar?
            // Weights are usually laid out contiguously per expert?
            // inputs.B matches [Experts * N * K]?
            // Check stride.
            // arguments constructor uses `ptr_B`.
            // In MoeFCGemm, `byte_ptr_B` calculation: `problem_idx * bytes_per_expert_matrix`.
            // So contiguous.
            const float* b_ptr = reinterpret_cast<const float*>(inputs.B) + e * inputs.n * inputs.k;

            float b_sample = 0;
            cudaMemcpy(&b_sample, b_ptr, sizeof(float), cudaMemcpyDeviceToHost);

            // printf("DEBUG [GEMM float] Expert %d: A[%lld]=%f, B[0]=%f\n", e, (long long)(start_row * inputs.k), a_sample, b_sample);
          } else {
            // printf("DEBUG [GEMM float] Expert %d: NO VALID ROWS\n", e);
          }
        }
        fflush(stdout);
      }
#endif

      GemmGrouped gemm;

      auto can_implement = gemm.can_implement(args);
      ORT_ENFORCE(can_implement == cutlass::Status::kSuccess,
                  "MoE FC kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement)));

      auto init_status = gemm.initialize(args);
      ORT_ENFORCE(init_status == cutlass::Status::kSuccess,
                  "Failed to initialize cutlass grouped gemm. Error: " + std::string(cutlassGetStatusString(init_status)));

      auto run_status = gemm.run(inputs.stream);
      ORT_ENFORCE(run_status == cutlass::Status::kSuccess,
                  "Failed to run cutlass grouped gemm. Error: " + std::string(cutlassGetStatusString(run_status)));
#if ORT_DEBUG_MOE
      // Debug: Sample output after GEMM for float types
      if constexpr (std::is_same_v<T, float>) {
        cudaStreamSynchronize(inputs.stream);

        // Re-use offsets copied earlier
        std::vector<int64_t> host_offsets(inputs.num_experts);
        cudaMemcpy(host_offsets.data(), inputs.total_tokens_including_expert, inputs.num_experts * sizeof(int64_t), cudaMemcpyDeviceToHost);

        for (int e = 0; e < inputs.num_experts; ++e) {
          int64_t start_row = (e == 0) ? 0 : host_offsets[e - 1];
          int64_t end_row = host_offsets[e];
          int64_t valid_rows = end_row - start_row;

          if (valid_rows > 0) {
            // C is [TotalRows, N]
            // ptr = C + start_row * N
            // inputs.C is void*? Cast to float*.
            const float* c_ptr = reinterpret_cast<const float*>(inputs.C) + start_row * inputs.n;
            float c_sample = 0;
            cudaMemcpy(&c_sample, c_ptr, sizeof(float), cudaMemcpyDeviceToHost);
            // printf("DEBUG [GEMM float] Expert %d: C[%lld]=%f (after GEMM)\n", e, (long long)(start_row * inputs.n), c_sample);
          }
        }
        fflush(stdout);
      }
#endif
    } else if constexpr (sizeof(ElementType) == 2 && sizeof(CutlassWeightType) == 2 && (std::is_same_v<EpilogueTag, cutlass_extensions::EpilogueOpDefaultSilu> || std::is_same_v<EpilogueTag, cutlass_extensions::EpilogueOpDefaultFtGelu>))  // use fused moe gemm
                                                                                                                                                                                                                                              // kernel.. (only support
                                                                                                                                                                                                                                              // fp16 or bf16)
    {
#ifdef EXCLUDE_SM_80
      ORT_THROW("Fused MoE SM80 kernels are not available because SM_80 was excluded from CMAKE_CUDA_ARCHITECTURES.");
#else
#ifdef ORT_QUICK_BUILD
      // Under QUICK_BUILD, BF16 fused MoE SM80 kernels are not instantiated.
      if constexpr (std::is_same_v<ElementType, cute::bfloat16_t>) {
        ORT_THROW("BF16 fused MoE SM80 kernels are not available under ORT_QUICK_BUILD.");
      } else {
#endif
        sm80_generic_fused_moe_gemm_kernelLauncher<ElementType, CutlassWeightType, ThreadblockShape::kM,
                                                   ThreadblockShape::kN, ThreadblockShape::kK, Stages, EpilogueTag>(
            reinterpret_cast<ElementType const*>(inputs.A), reinterpret_cast<CutlassWeightType const*>(inputs.B),
            reinterpret_cast<ElementType const*>(inputs.biases), inputs.bias_is_broadcast,
            reinterpret_cast<ElementType*>(inputs.C), inputs.total_tokens_including_expert, inputs.num_rows,
            inputs.n, inputs.k, inputs.num_experts, sm_count_, inputs.stream, inputs.occupancy);
#ifdef ORT_QUICK_BUILD
      }
#endif
#endif  // EXCLUDE_SM_80
    }
  }
};

template <typename GemmOutputType, typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag,
          typename ThreadblockShape, typename WarpShape, int Stages>
struct genericMoeGemmKernelLauncher<__nv_bfloat16, __nv_fp8_e4m3, GemmOutputType, arch, QuantOp, EpilogueTag,
                                    ThreadblockShape, WarpShape, Stages> {
  static void call(
      GroupedGemmInput<__nv_bfloat16, __nv_fp8_e4m3, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  }
};

// W8A16-FP8: half activations with FP8 weights — SM80 path is not supported, only SM90 TMA WS.
template <typename GemmOutputType, typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag,
          typename ThreadblockShape, typename WarpShape, int Stages>
struct genericMoeGemmKernelLauncher<half, __nv_fp8_e4m3, GemmOutputType, arch, QuantOp, EpilogueTag,
                                    ThreadblockShape, WarpShape, Stages> {
  static void call(
      GroupedGemmInput<half, __nv_fp8_e4m3, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  }
};

template <typename T, typename WeightType, typename GemmOutputType, typename Arch, typename EpilogueTag,
          typename ThreadblockShape, typename WarpShape, int Stages>
static void dispatch(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  ORT_LLM_LOG_ENTRY();
  static_assert(Arch::kMinComputeCapability < 90, "Use TMA specialized functions for arch SM90+");
#if defined(ENABLE_FP8)
  constexpr bool isFp8 = std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;
#else
  constexpr bool isFp8 = false;
#endif
#if defined(ENABLE_FP4)
  constexpr bool isFp4 = std::is_same_v<T, __nv_fp4_e2m1>;
#else
  constexpr bool isFp4 = false;
#endif

  if constexpr ((Stages == 2 || Arch::kMinComputeCapability >= 80) && (!isFp8 || std::is_same_v<Arch, cutlass::arch::Sm89>) && !isFp4) {
    // dispatch for quant op type
    auto* launcher = kernels::cutlass_kernels::genericMoeGemmKernelLauncher<T, WeightType, GemmOutputType, Arch,
                                                                            cutlass::WeightOnlyQuantOp::UNDEFINED, EpilogueTag, ThreadblockShape, WarpShape, Stages>::call;
    if (!std::is_same_v<WeightType, T> && inputs.groupwise_quant_group_size > 0) {
      launcher = inputs.zeros ? kernels::cutlass_kernels::genericMoeGemmKernelLauncher<T, WeightType,
                                                                                       GemmOutputType, Arch, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, EpilogueTag,
                                                                                       ThreadblockShape, WarpShape, Stages>::call
                              : kernels::cutlass_kernels::genericMoeGemmKernelLauncher<T, WeightType,
                                                                                       GemmOutputType, Arch, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
                                                                                       EpilogueTag, ThreadblockShape, WarpShape, Stages>::call;
    }
    launcher(inputs, sm_count_);
  } else {
    ORT_THROW(
        "Cutlass gemm. Not instantiated for arch %d with stages set to %d", Arch::kMinComputeCapability, Stages);
  }
}

template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
          typename ThreadblockShape, typename WarpShape,
          typename std::enable_if<(std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value) && !std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchGemmConfig(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  ORT_LLM_LOG_ENTRY();
  switch (inputs.gemm_config.stages) {
    case 2:
      dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>(inputs, sm_count_);
      break;
    case 3:
      dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>(inputs, sm_count_);
      break;
    case 4:
      dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>(inputs, sm_count_);
      break;
    default:
      ORT_THROW("dispatchGemmConfig does not support stages %d", inputs.gemm_config.stages);
      break;
  }
}

template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
          typename ThreadblockShape, typename WarpShape,
          typename std::enable_if<((!std::is_same<T, __nv_fp8_e4m3>::value) && (!std::is_same<T, __nv_fp8_e5m2>::value)) || std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchGemmConfig(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  ORT_LLM_LOG_ENTRY();
  switch (inputs.gemm_config.stages) {
#ifndef ORT_QUICK_BUILD
    case 2:
      dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(inputs, sm_count_);
      break;
    case 3:
      dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>(inputs, sm_count_);
      break;
#endif
    case 4:
      dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>(inputs, sm_count_);
      break;
    default:
      ORT_THROW("dispatchGemmConfig does not support stages %d", inputs.gemm_config.stages);
      break;
  }
}

// This overload will handle tensorop gemms. It is disabled via SFINAE for fp32.
// This overload is only enabled when T == WeightType.
template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
          typename std::enable_if<!std::is_same<T, float>::value
#if defined(ENABLE_FP8)
                                  && !std::is_same<T, __nv_fp8_e4m3>::value && !std::is_same<T, __nv_fp8_e5m2>::value
#endif
                                  && std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  ORT_LLM_LOG_ENTRY();
  switch (inputs.gemm_config.tile_config_sm80) {
#ifndef ORT_QUICK_BUILD
    case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
                           cutlass::gemm::GemmShape<16, 32, 64>>(inputs, sm_count_);
      }
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
                           cutlass::gemm::GemmShape<16, 64, 64>>(inputs, sm_count_);
      }
      break;
#endif
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<32, 64, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      ORT_THROW("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      ORT_THROW("GEMM config should have already been set by heuristic.");
      break;
    default:
      ORT_THROW("Config is invalid for same type tensorop GEMM.");
      break;
  }
}

// Tensorop GEMM overload
// Overload for quantize MoE GEMMs. We disable some warp configs here since they will not be used and we can improve
// compile time
template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
          typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  ORT_LLM_LOG_ENTRY();
  constexpr int tile_shape_k = 128 * 8 / cutlass::sizeof_bits<T>::value;
  switch (inputs.gemm_config.tile_config_sm80) {
#ifndef ORT_QUICK_BUILD
    case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag,
                           cutlass::gemm::GemmShape<16, 128, tile_shape_k>, cutlass::gemm::GemmShape<16, 32, tile_shape_k>>(
            inputs, sm_count_);
      }
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag,
                           cutlass::gemm::GemmShape<16, 256, tile_shape_k>, cutlass::gemm::GemmShape<16, 64, tile_shape_k>>(
            inputs, sm_count_);
      }
      break;
#endif
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag,
                         cutlass::gemm::GemmShape<32, 128, tile_shape_k>, cutlass::gemm::GemmShape<32, 32, tile_shape_k>>(
          inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag,
                         cutlass::gemm::GemmShape<64, 128, tile_shape_k>, cutlass::gemm::GemmShape<64, 32, tile_shape_k>>(
          inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag,
                         cutlass::gemm::GemmShape<128, 128, tile_shape_k>, cutlass::gemm::GemmShape<128, 32, tile_shape_k>>(
          inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      ORT_THROW("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      ORT_THROW("GEMM config should have already been set by heuristic.");
      break;
    default:
      ORT_THROW("Config is invalid for mixed type tensorop GEMM.");
      break;
  }
}

// This overload will handle tensorop gemms.
// This overload is only enabled when T == WeightType and T == __nv_fp8_e4m3 or __nv_fp8_e5m2
#if defined(ENABLE_FP8)
template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
          typename std::enable_if<(std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value) && std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  ORT_LLM_LOG_ENTRY();
  switch (inputs.gemm_config.tile_config_sm80) {
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 128>,
                         cutlass::gemm::GemmShape<16, 64, 128>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                         cutlass::gemm::GemmShape<32, 32, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 64, 128>,
                         cutlass::gemm::GemmShape<32, 64, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 64, 64>,
                         cutlass::gemm::GemmShape<64, 32, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 256, 64>,
                         cutlass::gemm::GemmShape<64, 64, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<256, 128, 64>,
                         cutlass::gemm::GemmShape<64, 64, 64>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      ORT_THROW("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      ORT_THROW("GEMM config should have already been set by heuristic.");
      break;
    default:
      ORT_THROW("Config is invalid for same type tensorop GEMM.");
      break;
  }
}
#endif

// This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
          typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs, int sm_count_) {
  ORT_LLM_LOG_ENTRY();
  switch (inputs.gemm_config.tile_config_sm80) {
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
      dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 8>,
                         cutlass::gemm::GemmShape<64, 64, 8>>(inputs, sm_count_);
      break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
      ORT_THROW("GEMM config undefined.");
      break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
      ORT_THROW("GEMM config should have already been set by heuristic.");
      break;
    default:
      ORT_THROW("Unsupported config for float MoE gemm.");
      break;
  }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getConfigs() const {
  ORT_LLM_LOG_ENTRY();
  return getConfigs(sm_);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig> MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getConfigs(
    int sm) {
  ORT_LLM_LOG_ENTRY();
  std::vector<cutlass_extensions::CutlassGemmConfig> candidate_configs = getTmaWarpSpecializedConfigs(sm);
  std::vector<cutlass_extensions::CutlassGemmConfig> ampere_configs = getAmpereConfigs(sm);
  std::copy(ampere_configs.begin(), ampere_configs.end(), std::back_inserter(candidate_configs));
  return candidate_configs;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getAmpereConfigs(int sm) {
  ORT_LLM_LOG_ENTRY();
  using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;
  static constexpr auto weight_only_flag = std::is_same<T, WeightType>::value ? CutlassGemmConfig::NONE : CutlassGemmConfig::WEIGHT_ONLY;
  static constexpr auto simt_only_flag = std::is_same<T, float>::value ? CutlassGemmConfig::SIMT_ONLY : CutlassGemmConfig::NONE;
  static constexpr auto fp8_only_flag = use_fp8 ? CutlassGemmConfig::FP8_ONLY : CutlassGemmConfig::NONE;
  int const max_split_k = 1;
  int const grouped_gemm_flag = CutlassGemmConfig::GROUPED_GEMM;
  int const enable_hopper = CutlassGemmConfig::NONE;

  auto config_type_param = static_cast<CutlassGemmConfig::CandidateConfigTypeParam>(
      weight_only_flag | simt_only_flag | grouped_gemm_flag | enable_hopper | fp8_only_flag);

  if (!kernels::cutlass_kernels::isValidAmpereMOESpecialisation<T, WeightType>() || (use_w4afp8 && sm != 89)) {
    return {};
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> ampere_configs = kernels::cutlass_kernels::get_candidate_configs(sm, max_split_k, config_type_param);
  return ampere_configs;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getTmaWarpSpecializedConfigs(int sm) {
  ORT_LLM_LOG_ENTRY();
  using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;
  static constexpr auto weight_only_flag = std::is_same<T, WeightType>::value ? CutlassGemmConfig::NONE : CutlassGemmConfig::WEIGHT_ONLY;
  static constexpr auto simt_only_flag = std::is_same<T, float>::value ? CutlassGemmConfig::SIMT_ONLY : CutlassGemmConfig::NONE;
  int const max_split_k = 1;
  int const grouped_gemm_flag = CutlassGemmConfig::GROUPED_GEMM;
  int const config_sm = use_wfp4a16 && sm >= 120 ? 90 : sm;
  int const enable_blackwell = config_sm >= 100 ? CutlassGemmConfig::BLACKWELL : CutlassGemmConfig::NONE;
  int const enable_hopper = config_sm == 90 ? CutlassGemmConfig::HOPPER : CutlassGemmConfig::NONE;
  static constexpr auto fp8_only_flag = use_fp8 ? CutlassGemmConfig::FP8_ONLY : CutlassGemmConfig::NONE;
  static constexpr auto fp4_only_flag = (use_fp4 || use_wfp4afp8) ? CutlassGemmConfig::FP4_ONLY : CutlassGemmConfig::NONE;
  auto config_type_param = static_cast<CutlassGemmConfig::CandidateConfigTypeParam>(weight_only_flag | simt_only_flag | grouped_gemm_flag | enable_blackwell | enable_hopper | fp8_only_flag | fp4_only_flag);
  ORT_ENFORCE(!(enable_blackwell && enable_hopper), "Blackwell and hopper flags are mutually exclusive");

  if (!isTmaWarpSpecializedGroupedGemmCompiledForSm(config_sm)) {
    if constexpr (use_w4afp8 || use_wfp4a16 || use_wfp4afp8 || use_wfp8a16 ||
                  use_fp4) {
      ORT_THROW(
          "TMA WS grouped MoE GEMM for SM%d is not compiled, and this QMoE configuration has no SM80 fallback",
          config_sm);
    }
    ORT_LLM_LOG_DEBUG(
        "TMA WS grouped MoE GEMM is not compiled for this SM, not selecting any TMA WS implementations");
    return {};
  }

  if (config_sm >= 100 && config_sm < 120 && !kernels::cutlass_kernels::isValidBlackwellMOESpecialisation<T, WeightType>()) {
    ORT_LLM_LOG_DEBUG("Blackwell is not supported for this configuration, not selecting any TMA WS implementations");
    return {};
  }
  if ((config_sm == 120 || config_sm == 121) && !kernels::cutlass_kernels::isValidSM120MOESpecialisation<T, WeightType>()) {
    ORT_LLM_LOG_DEBUG(
        "Blackwell SM120 is not supported for this configuration, not selecting any TMA WS implementations");
    return {};
  }
  if (enable_hopper && !kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType>()) {
    ORT_LLM_LOG_DEBUG("Hopper is not supported for this configuration, not selecting any TMA WS implementations");
    return {};
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> tma_ws_configs = kernels::cutlass_kernels::get_candidate_configs(config_sm, max_split_k, config_type_param);
  return tma_ws_configs;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::isTmaWarpSpecialized(
    cutlass_extensions::CutlassGemmConfig gemm_config) const {
  bool config_is_tma_warp_specialized = gemm_config.is_tma_warp_specialized;
  return supportsTmaWarpSpecialized() && config_is_tma_warp_specialized;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::supportsTmaWarpSpecialized() const {
  ORT_LLM_LOG_ENTRY();
  int const config_sm = use_wfp4a16 && sm_ >= 120 ? 90 : sm_;
  if (!isTmaWarpSpecializedGroupedGemmCompiledForSm(config_sm)) {
    return false;
  }

  if constexpr (use_wfp4a16) {
    return sm_ >= 120 && kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType>();
  } else {
    return (sm_ == 90 && kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType>()) ||
           (sm_ >= 100 && sm_ < 120 && kernels::cutlass_kernels::isValidBlackwellMOESpecialisation<T, WeightType>()) ||
           ((sm_ == 120 || sm_ == 121) && kernels::cutlass_kernels::isValidSM120MOESpecialisation<T, WeightType>());
  }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
int MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getSM() const {
  return this->sm_;
}

// currently support sm80 bf16/fp16 gate activation, only set predication tensor for m direction
template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::supportsFusedGatedActivation(
    bool is_gated_activation, int gemm_n, int gemm_k) const {
  // Fused gated activation (Ampere style) is NOT supported for gated activations (SwiGLU, GeGLU).
  // The Sm80 kernel dispatches to EpilogueOpDefault which is identity, not a fused gated epilogue.
  // For gated activations, we need the separate doGatedActivationKernel path:
  // - GEMM outputs 2N to intermediate buffer
  // - doGatedActivationKernel reduces 2N -> N
  // Setting this to false ensures fc1_out_size = 2*inter_size for gated activations.
  constexpr bool ENABLE_FUSED_GATED_ACTIVATION = false;
  return is_gated_activation && std::is_same_v<T, WeightType> && !std::is_same_v<T, float> && !use_fp8 &&
         (this->getSM() >= 80) && (gemm_k % 64 == 0) && (gemm_n % 64 == 0) && ENABLE_FUSED_GATED_ACTIVATION;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::isFusedGatedActivation(
    cutlass_extensions::CutlassGemmConfig gemm_config, bool is_gated_activation, int gemm_n, int gemm_k) const {
  return supportsFusedGatedActivation(is_gated_activation, gemm_n, gemm_k) && !gemm_config.is_tma_warp_specialized;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::MoeGemmRunner() {
  int device{-1};
  CUDA_CALL_THROW(cudaGetDevice(&device));
  sm_ = onnxruntime::llm::common::getSMVersion();
  CUDA_CALL_THROW(
      cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::dispatchToArch(
    GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
  ORT_LLM_LOG_ENTRY();
  static_assert(std::is_same_v<ScaleBiasType, OutputType>,
                "Separate Scale/Bias type is not supported. This is assumed to be the gemm output type");

  // For now we always cast this to output type.
  // In the future this will vary based on what fusions are applied for FP8
  // auto* C = reinterpret_cast<OutputType*>(C_void);

  ORT_ENFORCE(
      sm_ >= 89 || !hopper_inputs.isValid(), "Hopper input information is set for non specialized implementation");
  ORT_ENFORCE(sm_ >= 90 || !inputs.gemm_config.is_tma_warp_specialized,
              "Hopper configuration provided for non-Hopper architecture");

  /*if (sm_ >= 75 && sm_ < 80) {
    dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm75, EpilogueTag>(
        inputs, multi_processor_count_);
  } else*/
  if (sm_ >= 80 && sm_ < 90) {
    if constexpr (use_fp8 || use_w4afp8) {
#if defined(ENABLE_FP8)
      static_assert(!std::is_same_v<OutputType, __nv_fp8_e4m3> && !std::is_same_v<OutputType, __nv_fp8_e5m2>,
                    "FP8 GEMM Output not supported");
#endif

      ORT_ENFORCE(sm_ == 89, "For sm >= 80 and < 90, fp8 is only supported with sm == 89");
      dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm89, EpilogueTag>(
          inputs, multi_processor_count_);
    } else if constexpr (use_wfp4a16) {
      ORT_THROW("wfp4a16 (FP4 weights with FP16/BF16 activations) requires SM120+");
    } else if constexpr (use_wfp4afp8) {
      ORT_THROW("wfp4afp8 (FP4 weights with FP8 activations) requires SM100+");
    } else if constexpr (use_wfp8a16) {
      ORT_THROW("wfp8a16 (FP8 weights with FP16/BF16 activations) requires SM90+");
    } else if constexpr (use_fp4) {
      ORT_THROW("FP4 MoE GEMM requires SM90+");
    } else {
      dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm80, EpilogueTag>(
          inputs, multi_processor_count_);
    }
  } else if (sm_ >= 90) {
    // For SM120+ FP8 MoE, redirect to SM89 (Ada) FP8 kernel implementations.
    if constexpr (use_fp8) {
      if (sm_ >= 120) {
        dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm89, EpilogueTag>(
            inputs, multi_processor_count_);
        return;
      }
    }

    if constexpr (!std::is_same_v<std::decay_t<T>, float> && kernels::cutlass_kernels::isValidTmaWarpSpecializedMOESpecialisation<T, WeightType, EpilogueTag>() && !use_w4afp8 && !use_wfp4a16 && !use_wfp8a16) {
      // We allow both tma warp specialized and SM80 configurations to coexist because for some cases with small
      // numbers of tokens SM80 is faster. We check here to see which is selected
      if (inputs.gemm_config.sm_version >= 90) {
        ORT_ENFORCE(inputs.gemm_config.sm_version == sm_, "Using SM %d configuration for SM %d device",
                    inputs.gemm_config.sm_version, sm_);
        ORT_ENFORCE(inputs.biases != nullptr || hopper_inputs.ptr_c == nullptr,
                    "Input biases and hopper input disagree if bias is enabled");
        ORT_ENFORCE(
            hopper_inputs.isValid(), "Calling TMA warp specialized configuration with invalid hopper config");

        // Select the appropriate fusion function
        auto select_function = [&]() {
          switch (hopper_inputs.fusion) {
            case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE:
              return &dispatchMoeGemmSelectTileShapeTmaWarpSpecialized<T, WeightType, OutputType, EpilogueTag,
                                                                       TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE>;
            case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE:
              return &dispatchMoeGemmSelectTileShapeTmaWarpSpecialized<T, WeightType, OutputType, EpilogueTag,
                                                                       TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>;
            case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::ACTIVATION:
            case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION:
            default:
              ORT_THROW("Unimplemented fusion %d requested", (int)hopper_inputs.fusion);
          };
        };
        auto selected_func = select_function();
        selected_func(hopper_inputs, inputs.num_experts, inputs.gemm_config, multi_processor_count_,
                      inputs.stream, inputs.occupancy, nullptr);
        return;
      }

      // Fallthrough to SM80 impl below
    }

#if defined(ENABLE_FP8)
    // Hopper finegrained INT4 WS grouped GEMM
    if constexpr (use_w4afp8) {
      if (inputs.gemm_config.is_tma_warp_specialized) {
        // EpilogueTag is ignored
        if (inputs.k % 512 == 0) {
          sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<T, WeightType, ScaleBiasType,
                                                        cutlass_extensions::EpilogueOpDefault, 4>(
              inputs, hopper_inputs, multi_processor_count_, nullptr);
        } else if (inputs.k % 256 == 0) {
          sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<T, WeightType, ScaleBiasType,
                                                        cutlass_extensions::EpilogueOpDefault, 2>(
              inputs, hopper_inputs, multi_processor_count_, nullptr);
        } else if (inputs.k % 128 == 0) {
          sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<T, WeightType, ScaleBiasType,
                                                        cutlass_extensions::EpilogueOpDefault, 1>(
              inputs, hopper_inputs, multi_processor_count_, nullptr);
        } else {
          ORT_THROW("Invalid GEMM K size %d", (int)inputs.k);
        }
        return;
      };
    }
#endif

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
    // Hopper W4A16 (FP4 weights + FP16/BF16 activations) WS grouped GEMM
    if constexpr (use_wfp4a16) {
#ifdef ORT_QUICK_BUILD
      // Quick build only instantiates FP16+FP4 kernels; BF16+FP4 is not available.
      if constexpr (!std::is_same_v<T, half>) {
        ORT_THROW("BF16+FP4 MoE GEMM is not available under ORT_QUICK_BUILD. Use FP16 activations instead.");
      } else {
#endif
        ORT_ENFORCE(inputs.gemm_config.is_tma_warp_specialized,
                    "wfp4a16 is only supported for TMA warp specialization");
        // Select fusion and K tile based on runtime information
        auto select_fusion = [&]() {
          switch (hopper_inputs.fusion) {
            case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE:
              if (inputs.k % 256 == 0) {
                sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<T, WeightType, ScaleBiasType,
                                                              cutlass_extensions::EpilogueOpDefault,
                                                              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE, 1>(
                    inputs, hopper_inputs, multi_processor_count_, nullptr);
              } else {
                sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<T, WeightType, ScaleBiasType,
                                                              cutlass_extensions::EpilogueOpDefault,
                                                              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE, 2>(
                    inputs, hopper_inputs, multi_processor_count_, nullptr);
              }
              break;
            case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE:
            default:
              if (inputs.k % 256 == 0) {
                sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<T, WeightType, ScaleBiasType,
                                                              cutlass_extensions::EpilogueOpDefault,
                                                              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE, 1>(
                    inputs, hopper_inputs, multi_processor_count_, nullptr);
              } else {
                sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<T, WeightType, ScaleBiasType,
                                                              cutlass_extensions::EpilogueOpDefault,
                                                              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE, 2>(
                    inputs, hopper_inputs, multi_processor_count_, nullptr);
              }
              break;
          }
        };
        select_fusion();
        return;
#ifdef ORT_QUICK_BUILD
      }
#endif
    }
#endif

#if defined(ENABLE_FP8)
    // Hopper W8A16 (FP8 weights + FP16/BF16 activations) TMA WS grouped GEMM
    // Uses the same-type TMA WS path with mixed-precision CollectiveBuilder.
    // Per-expert global scale is applied via alpha_scale_ptr_array in the epilogue.
    if constexpr (use_wfp8a16) {
      ORT_ENFORCE(inputs.gemm_config.is_tma_warp_specialized,
                  "wfp8a16 is only supported for TMA warp specialization on SM90");
      // Route through the same-type TMA WS dispatch which handles mixed <half/bf16, fp8_e4m3> via CollectiveBuilder
      auto select_function = [&]() {
        switch (hopper_inputs.fusion) {
          case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE:
            return &dispatchMoeGemmSelectTileShapeTmaWarpSpecialized<T, WeightType, OutputType, EpilogueTag,
                                                                     TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE>;
          case TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE:
            return &dispatchMoeGemmSelectTileShapeTmaWarpSpecialized<T, WeightType, OutputType, EpilogueTag,
                                                                     TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>;
          default:
            ORT_THROW("Unimplemented fusion %d requested for wfp8a16", (int)hopper_inputs.fusion);
        };
      };
      auto selected_func = select_function();
      selected_func(hopper_inputs, inputs.num_experts, inputs.gemm_config, multi_processor_count_,
                    inputs.stream, inputs.occupancy, nullptr);
      return;
    }
#endif

    // Do Ampere case instead
    if constexpr (kernels::cutlass_kernels::isValidAmpereMOESpecialisation<T, WeightType, EpilogueTag>()) {
      ORT_ENFORCE(!use_fp8, "No fallback FP8 implementation available");
      ORT_ENFORCE(use_w4afp8 || use_wfp4a16 || use_wfp8a16 || !hopper_inputs.isValid(),
                  "Non-specialized Hopper implementation is being rerouted to fallback implementation so input "
                  "information is not required");
      ORT_ENFORCE(!inputs.gemm_config.is_tma_warp_specialized,
                  "GEMM config is for SM90 configuration, but this configuration is not valid for Hppper");
      ORT_ENFORCE(inputs.gemm_config.sm_version == 80,
                  "Using SM %d configuration for SM80 fallback implementation", inputs.gemm_config.sm_version);
      if constexpr (use_fp8) {
        dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm89, EpilogueTag>(
            inputs, multi_processor_count_);
      } else {
        dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm80, EpilogueTag>(
            inputs, multi_processor_count_);
      }
    } else {
      ORT_THROW("Configuration expects SM80 but configuration is not supported by SM80 kernels");
    }
  } else {
    ORT_THROW("Arch unsupported for MoE GEMM");
  }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
size_t MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getMaxWorkspaceSize(int num_experts) const {
  if (num_experts != num_experts_) {
    ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Calling getMaxWorkspaceSize() with a new expert count ", num_experts, " vs ", num_experts_));
    num_experts_ = num_experts;
    gemm_workspace_size_ = calcMaxWorkspaceSize(num_experts);
  }
  return gemm_workspace_size_;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
size_t MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::calcMaxWorkspaceSize(int num_experts) const {
  if constexpr (use_w4afp8 || use_wfp4a16) {
    return calcMaxWorkspaceSizeTmaWarpSpecializedMixedInput<T, WeightType, OutputType>(
        num_experts, multi_processor_count_);
  }
  if (!supportsTmaWarpSpecialized()) {
    return 0;
  }
  if constexpr (!std::is_same_v<std::decay_t<T>, float> && kernels::cutlass_kernels::isValidTmaWarpSpecializedMOESpecialisation<T, WeightType>() && !use_w4afp8 && !use_wfp4a16) {
    auto configs = getTmaWarpSpecializedConfigs(sm_);
    auto fpX_block_scaling_type = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE;
    if constexpr (use_wfp4afp8) {
      fpX_block_scaling_type = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
    } else if (use_fp4) {
      fpX_block_scaling_type = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
    }
    size_t max_size = 0;
    bool has_config = false;
    for (auto conf : configs) {
#define CALC_SIZE_FUSION(FUSION)                                                                                               \
  do {                                                                                                                         \
    try {                                                                                                                      \
      size_t size = calcMaxWorkspaceSizeTmaWarpSpecialized<T, WeightType, OutputType, FUSION>(                                 \
          num_experts, conf, multi_processor_count_, fpX_block_scaling_type);                                                  \
      max_size = std::max(max_size, size);                                                                                     \
      has_config = true;                                                                                                       \
    } catch (::onnxruntime::OnnxRuntimeException const& e) {                                                                   \
      ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Unsupported config skipped when calculating MOE workspace size ", e.what())); \
    }                                                                                                                          \
  } while (0)

      CALC_SIZE_FUSION(TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE);
      if (sm_ == 90) {
        CALC_SIZE_FUSION(TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE);
      }

#undef CALC_SIZE_FUSION
    }
    ORT_ENFORCE(has_config, "Could not find valid config when calculating workspace size");
    return max_size;
  } else {
    ORT_THROW("Attempting to calculate Hopper GEMM workspace size with unsupported weight combination");
  }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::runGemm(
    GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
  dispatchToArch<EpilogueTag>(inputs, hopper_inputs);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::moeGemmBiasAct(
    GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType>
        inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
  ORT_LLM_LOG_ENTRY();
  switch (inputs.activation_type) {
    case ActivationType::Relu:
      runGemm<cutlass_extensions::EpilogueOpDefaultReLU>(inputs, hopper_inputs);
      break;
    case ActivationType::Gelu:
      runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(inputs, hopper_inputs);
      break;
    case ActivationType::Silu:
      runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(inputs, hopper_inputs);
      break;
    case ActivationType::Identity:
      runGemm<cutlass_extensions::EpilogueOpDefault>(inputs, hopper_inputs);
      break;
    case ActivationType::Swiglu:
      // Match TRT-LLM: use SiLu epilogue for fused path
      runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(inputs, hopper_inputs);
      break;
    case ActivationType::Geglu:
      runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(inputs, hopper_inputs);
      break;
    case ActivationType::SwigluBias:
      // SwigluBias uses SiLu with per-expert alpha/beta/limit
      runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(inputs, hopper_inputs);
      break;
    case ActivationType::Relu2:
      runGemm<cutlass_extensions::EpilogueOpDefaultReLU>(inputs, hopper_inputs);
      break;
    case ActivationType::InvalidType:
      ORT_THROW("Activation type for fpA_intB must be valid.");
      break;
    default:
      ORT_THROW("Invalid activation type.");
      break;
  }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::moeGemm(
    GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType>
        inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
  ORT_LLM_LOG_ENTRY();
  runGemm<cutlass_extensions::EpilogueOpDefault>(inputs, hopper_inputs);
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
