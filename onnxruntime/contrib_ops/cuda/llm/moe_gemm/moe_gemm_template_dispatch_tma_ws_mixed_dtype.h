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
#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"

#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/compute_occupancy.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/threadblock/default_mma.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/cutlass_heuristic.h"
#include "core/common/common.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace onnxruntime::llm::kernels::cutlass_kernels {

namespace tk = onnxruntime::llm::common;
namespace tkc = onnxruntime::llm::cutlass_extensions;

using namespace cute;

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag, EpilogueFusion FUSION,
          typename CTAShape, typename ClusterShape>
void sm90_dispatch_mainloop_schedules(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
                                      TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  ORT_LLM_LOG_ENTRY();
#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
  switch (inputs.gemm_config.mainloop_schedule) {
#ifndef ORT_QUICK_BUILD
    case tkc::MainloopScheduleType::COOPERATIVE:
      if constexpr (get<0>(CTAShape{}) < 128) {
        ORT_THROW("COOPERATIVE is only enabled when tile M >= 128.");
      } else {
        if constexpr (
#if defined(ENABLE_FP4)
            std::is_same_v<WeightType, __nv_fp4_e2m1> &&
#else
            false &&
#endif
            std::is_same_v<T, half> && get<0>(CTAShape{}) == 128 && get<1>(CTAShape{}) == 32) {
          sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape,
                                                     ClusterShape, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                                                     cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(
              inputs, hopper_inputs, sm_count_, workspace_size);
        } else if constexpr ((get<0>(CTAShape{}) == 128) && get<1>(CTAShape{}) == 128) {
          sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape,
                                                     ClusterShape, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                                                     cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(
              inputs, hopper_inputs, sm_count_, workspace_size);
        } else {
          sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape,
                                                     ClusterShape, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                                                     cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(
              inputs, hopper_inputs, sm_count_, workspace_size);
        }
      }

      break;
#endif  // !ORT_QUICK_BUILD
    case tkc::MainloopScheduleType::PINGPONG:
      // fallthrough — AUTO uses PINGPONG which works for all tile sizes including M < 128.
    case tkc::MainloopScheduleType::AUTO:
      if constexpr (
#if defined(ENABLE_FP4)
          std::is_same_v<WeightType, __nv_fp4_e2m1> &&
#else
          false &&
#endif
          std::is_same_v<T, half> && get<0>(CTAShape{}) == 128 && get<1>(CTAShape{}) == 32) {
        sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape, ClusterShape,
                                                   cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative,
                                                   cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(inputs, hopper_inputs, sm_count_, workspace_size);
      } else if constexpr (
#if defined(ENABLE_FP4)
          std::is_same_v<WeightType, __nv_fp4_e2m1> &&
#else
          false &&
#endif
          get<0>(CTAShape{}) == 128 && (get<1>(CTAShape{}) == 32 || get<1>(CTAShape{}) == 64)) {
        sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape, ClusterShape,
                                                   cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative,
                                                   cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(inputs, hopper_inputs, sm_count_, workspace_size);
      } else {
        sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape, ClusterShape,
                                                   cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative,
                                                   cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(inputs, hopper_inputs, sm_count_, workspace_size);
      }
      break;
    default:
      ORT_THROW(
          "[Mixed dtype MoE GEMM][sm90_dispatch_mainloop_schedules] mainloop schedule config is invalid "
          "for "
          "mixed type GEMM.");
      break;
  }
#else
  ORT_THROW("Please recompile with support for hopper by passing 90-real as an arch to build_wheel.py.");
#endif
}

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag, EpilogueFusion FUSION,
          typename CTAShape>
void sm90_dispatch_moe_mixed_dtype_gemm_config(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
                                               TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  ORT_LLM_LOG_ENTRY();
  switch (inputs.gemm_config.cluster_shape) {
    case tkc::ClusterShape::ClusterShape_1x1x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape, Shape<_1, _1, _1>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
#ifndef ORT_QUICK_BUILD
    case tkc::ClusterShape::ClusterShape_2x1x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape, Shape<_2, _1, _1>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape, Shape<_1, _2, _1>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, FUSION, CTAShape, Shape<_2, _2, _1>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
#endif  // !ORT_QUICK_BUILD
    default:
      ORT_THROW("[Mixed dtype MoE GEMM][dispatch_CGA_config] Config is invalid for mixed type GEMM.");
      break;
  }
}

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag, EpilogueFusion FUSION, int PackedScalesNum>
void sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  ORT_LLM_LOG_ENTRY();
  // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually perform the best
  // for mixed type gemms.

#if defined(ENABLE_FP4)
  constexpr bool is_wfp4a16 = std::is_same_v<WeightType, __nv_fp4_e2m1>;
#else
  constexpr bool is_wfp4a16 = false;
#endif
  // For wfp4a16, K tile comes from the dispatch caller via PackedScalesNum encoding:
  //   PackedScalesNum == 1 → K=256, PackedScalesNum == 2 → K=128
  constexpr int Ktile = is_wfp4a16 ? (PackedScalesNum == 2 ? 128 : 256) : 128 * PackedScalesNum / sizeof(T);
  ORT_ENFORCE(sizeof(T) == (is_wfp4a16 ? 2 : 1));

  // For wfp4a16, no generated kernels for m64_n128 FP4 tiles; cap N at 64 for 64-row shapes.
  constexpr int Ntile = is_wfp4a16 ? 64 : 128;
  using _Ntile = Int<Ntile>;
  using _Ktile = Int<Ktile>;
  switch (inputs.gemm_config.tile_config_sm90) {
#ifndef ORT_QUICK_BUILD
    case tkc::CutlassTileConfigSM90::CtaShape64x16x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION, Shape<_64, _16, _Ktile>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape64x32x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION, Shape<_64, _32, _Ktile>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape64x64x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION, Shape<_64, _64, _Ktile>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape64x128x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION, Shape<_64, _Ntile, _Ktile>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
#endif  // !ORT_QUICK_BUILD
    case tkc::CutlassTileConfigSM90::CtaShape128x16x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION, Shape<_128, _16, _Ktile>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape128x32x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION, Shape<_128, _32, _Ktile>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape128x64x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION, Shape<_128, _64, _Ktile>>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape128x128x128B:
      sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag, FUSION,
                                                Shape<_128, _128, _Ktile>>(inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::CutlassTileConfigSM90::Undefined:
      ORT_THROW("[Mixed dtype MoE GEMM][sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass] gemm config undefined.");
      break;
    case tkc::CutlassTileConfigSM90::ChooseWithHeuristic:
      ORT_THROW(
          "[Mixed dtype MoE GEMM][sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass] gemm config should have already "
          "been set by "
          "heuristic.");
      break;
    default:
      ORT_THROW(
          "[Mixed dtype MoE GEMM][sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass] Config is invalid for mixed type "
          "GEMM.");
      break;
  }
}

template <typename T, typename WeightType, typename OutputType>
size_t calcMaxWorkspaceSizeTmaWarpSpecializedMixedInput(int num_experts, int sm_count_) {
  size_t count = 0;
#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
  constexpr int Ktile =
#if defined(ENABLE_FP4)
      (std::is_same_v<WeightType, __nv_fp4_e2m1>) ? 256 :
#endif
                                                  512;
  using _Ktile = Int<Ktile>;
  GroupedGemmInput<T, WeightType, OutputType, OutputType> inputs{};
  inputs.num_experts = num_experts;
  // Use cooperative kernel with m128_n64 tile for workspace calculation (launchers exist for all weight types).
  sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, OutputType,
                                             onnxruntime::llm::cutlass_extensions::EpilogueOpDefault, EpilogueFusion::NONE,
                                             Shape<_128, _64, _Ktile>, Shape<_1, _1, _1>,
                                             cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::epilogue::TmaWarpSpecializedCooperative,
                                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(
      inputs, TmaWarpSpecializedGroupedGemmInput{}, sm_count_, &count);
#endif
  return count;
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
