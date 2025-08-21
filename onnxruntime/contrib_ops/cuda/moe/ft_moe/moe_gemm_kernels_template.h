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

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

// Ignore CUTLASS warning C4100: unreferenced formal parameter
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"

#include "contrib_ops/cuda/moe/cutlass_extensions/compute_occupancy.h"
#include "contrib_ops/cuda/moe/cutlass_extensions/epilogue_helpers.h"
#include "contrib_ops/cuda/moe/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "contrib_ops/cuda/moe/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "contrib_ops/cuda/moe/cutlass_extensions/gemm/threadblock/default_mma.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "cutlass_heuristic.h"
#include "moe_gemm_kernels.h"

#include <cuda_bf16.h>

#include <limits>
#include <math.h>
#include <sstream>

namespace ort_fastertransformer {

// ============================= Variable batched Gemm things ===========================
template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape, int Stages>
void generic_moe_gemm_kernelLauncher(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                                     int64_t* total_rows_before_expert, int64_t gemm_n, int64_t gemm_k, int num_experts,
                                     CutlassGemmConfig gemm_config, const int multi_processor_count,
                                     cudaStream_t stream, int* kernel_occupancy = nullptr) {
  static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value || cutlass::platform::is_same<T, __nv_bfloat16>::value,
                "Specialized for half, float, bfloat16");

  static_assert(cutlass::platform::is_same<T, WeightType>::value ||
                    cutlass::platform::is_same<WeightType, uint8_t>::value ||
                    cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
                "");

  // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
  using ElementType_ =
      typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, typename cutlass::platform::conditional<cutlass::platform::is_same<T, __nv_bfloat16>::value, cutlass::bfloat16_t, T>::type>::type;
  using ElementType = ElementType_;

  using CutlassWeightType_ =
      typename cutlass::platform::conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t, typename cutlass::platform::conditional<cutlass::platform::is_same<WeightType, __nv_bfloat16>::value, cutlass::bfloat16_t, WeightType>::type>::type;

  using CutlassWeightType = CutlassWeightType_;

  // We need separate config for each architecture since we will target different tensorcore instructions. For
  // float, we do not target TCs.
  using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
  using ElementAccumulator = typename MixedGemmArchTraits::AccType;

  using EpilogueOp =
      typename Epilogue<ElementType, MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

  // Finally, set up the kernel.
  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementType, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA,
      CutlassWeightType, typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
      MixedGemmArchTraits::ElementsPerAccessB, ElementType, cutlass::layout::RowMajor, ElementAccumulator,
      typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
      typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;

  using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
                                                      typename GemmKernel_::ThreadblockSwizzle,
                                                      arch,  // Ensure top level arch is used for dispatch
                                                      GemmKernel_::kGroupScheduleMode>;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  if (kernel_occupancy != nullptr) {
    *kernel_occupancy = compute_occupancy_for_kernel<GemmKernel>();
    return;
  }
  int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
  ORT_ENFORCE(occupancy > 0, "GPU lacks the shared memory resources to run GroupedGEMM kernel");
  int const threadblock_count = multi_processor_count * occupancy;

  typename EpilogueOp::Params epilogue_op(ElementAccumulator(1.f),
                                          biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));

  int const group_size = gemm_k;
  typename GemmGrouped::Arguments args(
      num_experts, threadblock_count, group_size, epilogue_op, reinterpret_cast<ElementType const*>(A),
      reinterpret_cast<CutlassWeightType const*>(B), reinterpret_cast<ElementType const*>(weight_scales),
      reinterpret_cast<ElementType const*>(biases), reinterpret_cast<ElementType*>(C), total_rows_before_expert, gemm_n,
      gemm_k);

  // DEBUG: Print ALL GEMM calls to understand the pattern
  printf("[CUDA GEMM DEBUG] ALL CALLS: gemm_n=%ld, gemm_k=%ld, num_experts=%d\n", gemm_n, gemm_k, num_experts);

  // DEBUG: Print FC1 and FC2 GEMM input values
  bool is_fc1_gemm = (gemm_n >= 11000);                   // FC1: hidden_size -> inter_size (large gemm_n, e.g., 2880->11520)
  bool is_fc2_gemm = (gemm_k >= 5000 && gemm_n <= 8192);  // FC2: inter_size -> hidden_size (large gemm_k, small gemm_n, e.g., 5760->2880)

  if (is_fc1_gemm || is_fc2_gemm) {
    const char* gemm_type = is_fc1_gemm ? "FC1" : "FC2";
    printf("[CUDA %s GEMM DEBUG] Before GEMM: gemm_n=%ld, gemm_k=%ld, num_experts=%d\n", gemm_type, gemm_n, gemm_k, num_experts);

    // Print first few input values for Expert 5 (if exists) with bounds checking
    if (num_experts > 5 && total_rows_before_expert != nullptr) {
      // Safely check if Expert 5 has any rows assigned
      int64_t expert5_start = total_rows_before_expert[5];
      int64_t expert6_start = (num_experts > 6) ? total_rows_before_expert[6] : expert5_start;
      int64_t expert5_rows = expert6_start - expert5_start;

      printf("[CUDA %s GEMM DEBUG] total_rows_before_expert[5]=%ld, next=%ld\n",
             gemm_type, expert5_start, expert6_start);

      if (expert5_rows > 0 && expert5_start >= 0) {
        printf("[CUDA %s GEMM DEBUG] Expert 5 has %ld rows starting at %ld\n", gemm_type, expert5_rows, expert5_start);

        // Calculate input index with overflow checking
        int64_t input_start_idx = expert5_start * gemm_k;
        int64_t max_input_elements = 8;

        if (input_start_idx >= 0 && gemm_k > 0 && A != nullptr) {
          printf("[CUDA %s GEMM DEBUG] Expert 5 input A[%ld::%ld]: ", gemm_type, input_start_idx, input_start_idx + max_input_elements);
          for (int i = 0; i < max_input_elements && i < gemm_k; ++i) {
            printf("%.6f ", static_cast<float>(A[input_start_idx + i]));
          }
          printf("...\n");
        }

        // Print quantized weights B for Expert 5 (first few values) with bounds checking
        if (B != nullptr && gemm_k > 0 && gemm_n > 0) {
          // Calculate weight index with overflow protection
          int64_t weight_base_idx = 0;
          int64_t max_weight_elements = 8;

          if (is_fc1_gemm) {
            // For FC1: Expert 5 weights start at offset
            if (gemm_k <= INT32_MAX / gemm_n && gemm_n <= INT32_MAX / 2) {
              weight_base_idx = 5LL * (gemm_k * gemm_n / 2);
            }
          } else {
            // For FC2: Same calculation but different semantics
            if (gemm_k <= INT32_MAX / gemm_n && gemm_n <= INT32_MAX / 2) {
              weight_base_idx = 5LL * (gemm_k * gemm_n / 2);
            }
          }

          if (weight_base_idx >= 0) {
            printf("[CUDA %s GEMM DEBUG] Expert 5 weights B[%ld::%ld]: ", gemm_type, weight_base_idx, weight_base_idx + max_weight_elements);
            for (int i = 0; i < max_weight_elements; ++i) {
              if constexpr (std::is_same<CutlassWeightType, cutlass::uint4b_t>::value) {
                printf("0x%02x ", reinterpret_cast<const uint8_t*>(B)[weight_base_idx + i]);
              } else {
                // For non-4bit weights, adjust index calculation
                int64_t full_weight_idx = 5LL * gemm_k * gemm_n + i;
                if (full_weight_idx >= 0 && gemm_k <= INT32_MAX / gemm_n) {
                  printf("%.6f ", static_cast<float>(B[full_weight_idx]));
                }
              }
            }
            printf("...\n");
          }
        }

        // Print scales for Expert 5 with bounds checking
        if (weight_scales != nullptr && gemm_n > 0) {
          int64_t scales_base_idx = 5LL * gemm_n;
          int64_t max_scale_elements = (gemm_n < 8) ? gemm_n : 8;

          if (scales_base_idx >= 0) {
            printf("[CUDA %s GEMM DEBUG] Expert 5 scales[%ld::%ld]: ", gemm_type, scales_base_idx, scales_base_idx + max_scale_elements);
            for (int64_t i = 0; i < max_scale_elements; ++i) {
              printf("%.6f ", static_cast<float>(weight_scales[scales_base_idx + i]));
            }
            printf("...\n");
          }
        }
      }
    }
  }

  GemmGrouped gemm;

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg =
        "MoEFC kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
    ORT_THROW("[MoE Runner] " + err_msg);
  }

  auto init_status = gemm.initialize(args);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass variable batched gemm. Error: " +
                          std::string(cutlassGetStatusString(init_status));
    ORT_THROW("[MoE Runner] " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to run cutlass variable batched gemm. Error: " + std::string(cutlassGetStatusString(run_status));
    ORT_THROW("[MoE Runner] " + err_msg);
  }

  // DEBUG: Print FC1 and FC2 GEMM output values
  if (is_fc1_gemm || is_fc2_gemm) {
    cudaStreamSynchronize(stream);  // Ensure GEMM is complete before reading output

    const char* gemm_type = is_fc1_gemm ? "FC1" : "FC2";

    if (num_experts > 5 && total_rows_before_expert != nullptr && C != nullptr) {
      int64_t expert5_start = total_rows_before_expert[5];
      int64_t expert6_start = (num_experts > 6) ? total_rows_before_expert[6] : expert5_start;
      int64_t expert5_rows = expert6_start - expert5_start;

      if (expert5_rows > 0 && expert5_start >= 0 && gemm_n > 0) {
        // Calculate output index with overflow checking
        int64_t output_start_idx = expert5_start * gemm_n;
        int64_t max_output_elements = (gemm_n < 8) ? gemm_n : 8;

        if (output_start_idx >= 0) {
          // Print output values for Expert 5 first row
          printf("[CUDA %s GEMM DEBUG] Expert 5 output C[%ld::%ld]: ", gemm_type, output_start_idx, output_start_idx + max_output_elements);
          for (int64_t i = 0; i < max_output_elements; ++i) {
            printf("%.6f ", static_cast<float>(C[output_start_idx + i]));
          }
          printf("...\n");
        }
      }
    }
  }
}

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape, int Stages, typename Enable = void>
struct dispatch_stages {
  static void dispatch(const T* /*A*/, const WeightType* /*B*/, const T* /*weight_scales*/, const T* /*biases*/,
                       T* /*C*/, int64_t* /*total_rows_before_expert*/, int64_t /*gemm_n*/, int64_t /*gemm_k*/,
                       int /*num_experts*/, CutlassGemmConfig /*gemm_config*/, int /*multi_processor_count*/,
                       cudaStream_t /*stream*/, [[maybe_unused]] int* occupancy = nullptr) {
    std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch " +
                          std::to_string(arch::kMinComputeCapability) + " with stages set to " + std::to_string(Stages);
    ORT_THROW("[dispatch_stages::dispatch] " + err_msg);
  }
};

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape>
struct dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2> {
  static void dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                       int64_t* total_rows_before_expert, int64_t gemm_n, int64_t gemm_k, int num_experts,
                       CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_moe_gemm_kernelLauncher<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(
        A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts, gemm_config,
        multi_processor_count, stream, occupancy);
  }
};

template <typename T, typename WeightType, typename EpilogueTag, typename ThreadblockShape, typename WarpShape,
          int Stages>
struct dispatch_stages<T, WeightType, cutlass::arch::Sm80, EpilogueTag, ThreadblockShape, WarpShape, Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
  static void dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                       int64_t* total_rows_before_expert, int64_t gemm_n, int64_t gemm_k, int num_experts,
                       CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_moe_gemm_kernelLauncher<T, WeightType, cutlass::arch::Sm80, EpilogueTag, ThreadblockShape, WarpShape,
                                    Stages>(A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k,
                                            num_experts, gemm_config, multi_processor_count, stream, occupancy);
  }
};

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
          typename WarpShape>
void dispatch_gemm_config(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                          int64_t* total_rows_before_expert, int64_t gemm_n, int64_t gemm_k, int num_experts,
                          CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
                          int* occupancy = nullptr) {
  switch (gemm_config.stages) {
    case 2:
      using DispatcherStages2 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>;
      DispatcherStages2::dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
                                  gemm_config, multi_processor_count, stream, occupancy);
      break;
    case 3:
      using DispatcherStages3 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>;
      DispatcherStages3::dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
                                  gemm_config, multi_processor_count, stream, occupancy);
      break;
    case 4:
      using DispatcherStages4 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>;
      DispatcherStages4::dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
                                  gemm_config, multi_processor_count, stream, occupancy);
      break;
    default:
      std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
      ORT_THROW("[MoE][dispatch_gemm_config] " + err_msg);
      break;
  }
}

// This overload will handle tensorop gemms. It is disabled via SFINAE for fp32.
// This overload is only enabled when T == WeightType.
template <
    typename T, typename WeightType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                                  int64_t* total_rows_before_expert, int64_t /*total_rows*/, int64_t gemm_n,
                                  int64_t gemm_k, int num_experts, CutlassGemmConfig gemm_config, int /*sm_version*/,
                                  int multi_processor_count, cudaStream_t stream, int* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
                             cutlass::gemm::GemmShape<16, 32, 64>>(
            A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
            gemm_config, multi_processor_count, stream, occupancy);
      }
      break;
    case CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
                             cutlass::gemm::GemmShape<16, 64, 64>>(
            A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
            gemm_config, multi_processor_count, stream, occupancy);
      }
      break;
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                           cutlass::gemm::GemmShape<32, 32, 64>>(
          A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
          gemm_config, multi_processor_count, stream, occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                           cutlass::gemm::GemmShape<32, 64, 64>>(
          A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
          gemm_config, multi_processor_count, stream, occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                           cutlass::gemm::GemmShape<64, 32, 64>>(
          A, B, weight_scales, biases, C, total_rows_before_expert, gemm_n, gemm_k, num_experts,
          gemm_config, multi_processor_count, stream, occupancy);
      break;
    case CutlassTileConfig::Undefined:
      ORT_THROW("GEMM config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
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
template <
    typename T, typename WeightType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                                  int64_t* total_rows_before_expert, int64_t /*total_rows*/, int64_t gemm_n,
                                  int64_t gemm_k, int num_experts, CutlassGemmConfig gemm_config, int sm_version,
                                  int multi_processor_count, cudaStream_t stream, int* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
                             cutlass::gemm::GemmShape<16, 32, 64>>(
            A, B, weight_scales, biases, C, total_rows_before_expert,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
      }
      break;
    case CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
      ORT_ENFORCE(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
      if constexpr (arch::kMinComputeCapability >= 75) {
        dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
                             cutlass::gemm::GemmShape<16, 64, 64>>(
            A, B, weight_scales, biases, C, total_rows_before_expert,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
      }
      break;
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                           cutlass::gemm::GemmShape<32, 32, 64>>(
          A, B, weight_scales, biases, C, total_rows_before_expert,
          gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                           cutlass::gemm::GemmShape<64, 32, 64>>(
          A, B, weight_scales, biases, C, total_rows_before_expert,
          gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                           cutlass::gemm::GemmShape<128, 32, 64>>(
          A, B, weight_scales, biases, C, total_rows_before_expert,
          gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
      break;
    case CutlassTileConfig::Undefined:
      ORT_THROW("GEMM config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      ORT_THROW("GEMM config should have already been set by heuristic.");
      break;
    default:
      ORT_THROW("Config is invalid for mixed type tensorop GEMM.");
      break;
  }
}

// This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
template <typename T, typename WeightType, typename arch, typename EpilogueTag,
          typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void dispatch_moe_gemm_to_cutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
                                  int64_t* total_rows_before_expert, int64_t /*total_rows*/, int64_t gemm_n,
                                  int64_t gemm_k, int num_experts, CutlassGemmConfig gemm_config, int /*sm_version*/,
                                  int multi_processor_count, cudaStream_t stream, int* occupancy = nullptr) {
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
      dispatch_gemm_config<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 8>,
                           cutlass::gemm::GemmShape<64, 64, 8>>(
          A, B, weight_scales, biases, C, total_rows_before_expert,
          gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
      break;
    case CutlassTileConfig::Undefined:
      ORT_THROW("GEMM config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      ORT_THROW("GEMM config should have already been set by heuristic.");
      break;
    default:
      ORT_THROW("Unsupported config for float MoE gemm.");
      break;
  }
}

template <typename T, typename WeightType>
MoeGemmRunner<T, WeightType>::MoeGemmRunner() {}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::initialize(int sm_version) {
  int device{-1};
  cudaGetDevice(&device);
  sm_ = sm_version;
  cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device);
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag>(const T* A, const WeightType* B,
                                                                 const T* weight_scales, const T* biases, T* C,
                                                                 int64_t* total_rows_before_expert, int64_t total_rows,
                                                                 int64_t gemm_n, int64_t gemm_k, int num_experts,
                                                                 CutlassGemmConfig gemm_config, cudaStream_t stream,
                                                                 int* occupancy) {
  if (sm_ >= 70 && sm_ < 75) {
    dispatch_moe_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm70, EpilogueTag>(
        A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
        sm_, multi_processor_count_, stream, occupancy);
  } else if (sm_ >= 75 && sm_ < 80) {
    dispatch_moe_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm75, EpilogueTag>(
        A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
        sm_, multi_processor_count_, stream, occupancy);
  } else if (sm_ >= 80) {  // Hopper and Blackwell will fallback to use Ampere kernels.
    dispatch_moe_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(
        A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
        sm_, multi_processor_count_, stream, occupancy);
  }
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::profile_gemm<EpilogueTag>(const T* A, const WeightType* B, const T* weight_scales,
                                                             const T* biases, T* C, int64_t* total_rows_before_expert,
                                                             int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                                                             int num_experts, cudaStream_t stream, int64_t key) {
  static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
  static constexpr bool only_simt_configs = std::is_same<T, float>::value;

  std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(sm_, is_weight_only, only_simt_configs);
  std::vector<int> occupancies(candidate_configs.size());

  constexpr int warmup = 5;
  constexpr int runs = 10;
  float min_elapsed = std::numeric_limits<float>::max();
  size_t chosen_config_id = 0;
  for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
    for (int jj = 0; jj < warmup; ++jj) {
      dispatch_to_arch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n,
                                    gemm_k, num_experts, candidate_configs[ii], stream);
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);

    for (int jj = 0; jj < runs; ++jj) {
      dispatch_to_arch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n,
                                    gemm_k, num_experts, candidate_configs[ii], stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (elapsed < min_elapsed) {
      min_elapsed = elapsed;
      chosen_config_id = ii;
    }
  }
  CutlassGemmConfig config = candidate_configs[chosen_config_id];
  GetGemmConfigMap().Insert(key, config);
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::run_gemm<EpilogueTag>(const T* A, const WeightType* B, const T* weight_scales,
                                                         const T* biases, T* C, int64_t* total_rows_before_expert,
                                                         int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                                                         int num_experts, cudaStream_t stream) {
  // Generate Key to the GemmConfigMap
  // First 32 bits are total_rows, next 16 bits are gemm_n, next 16 bits are gemm_k
  int64_t key = total_rows;
  key = key << 16 | gemm_n;
  key = key << 16 | gemm_k;

  if (!GetGemmConfigMap().Contains(key)) {
    profile_gemm<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k,
                              num_experts, stream, key);
  }
  dispatch_to_arch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k,
                                num_experts, GetGemmConfigMap().Get(key), stream);
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm_bias_act(const T* A, const WeightType* B, const T* weight_scales,
                                                     const T* biases, T* C, int64_t* total_rows_before_expert,
                                                     int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
                                                     int num_experts, ActivationType activation_type,
                                                     cudaStream_t stream) {
  // Swiglu will use Identity to call this function so we not need to handle it here.
  switch (activation_type) {
    case ActivationType::Relu:
      run_gemm<EpilogueOpDefaultReLU>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n,
                                      gemm_k, num_experts, stream);
      break;
    case ActivationType::Gelu:
      run_gemm<EpilogueOpDefaultFtGelu>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n,
                                        gemm_k, num_experts, stream);
      break;
    case ActivationType::Silu:
      run_gemm<EpilogueOpDefaultSilu>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n,
                                      gemm_k, num_experts, stream);
      break;
    case ActivationType::Identity:
      run_gemm<EpilogueOpDefault>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k,
                                  num_experts, stream);
      break;
    case ActivationType::InvalidType:
      ORT_THROW("[MoE Runner] Invalid activation type for MoE GEMM");
      break;
    default: {
      ORT_THROW("[MoE Runner] Invalid activation type for MoE GEMM");
    }
  }
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moe_gemm(const T* A, const WeightType* B, const T* weight_scales, const T* biases,
                                            T* C, int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n,
                                            int64_t gemm_k, int num_experts, cudaStream_t stream) {
  run_gemm<EpilogueOpDefault>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k,
                              num_experts, stream);
}

}  // namespace ort_fastertransformer
