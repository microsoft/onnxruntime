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

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

#include "contrib_ops/cuda/llm/cutlass_extensions/compute_occupancy.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/threadblock/default_mma.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif

#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/cutlass_heuristic.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_launcher.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_tma_warp_specialized_traits.h"

#include "core/common/common.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace onnxruntime::llm::kernels::cutlass_kernels {
using EpilogueFusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion;

template <typename Arch, typename T, typename WeightType, typename OutputType, typename EpilogueTag,
          EpilogueFusion FUSION, typename TileShape, typename ClusterShape>
void dispatchMoeGemmSelectBiasTmaWarpSpecialized(TmaWarpSpecializedGroupedGemmInput hopper_input, int num_experts,
                                                 int multi_processor_count, cudaStream_t stream, int* occupancy, size_t* workspace_size) {
  static_assert((Arch::kMinComputeCapability == 90 && kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType, EpilogueTag>()) || (Arch::kMinComputeCapability >= 100 && kernels::cutlass_kernels::isValidBlackwellMOESpecialisation<T, WeightType, EpilogueTag>()),
                "Invalid TMA WS configuration invoked, fallback to Sm80");

  ORT_ENFORCE(
      workspace_size || hopper_input.isValid(), "Hopper specialisation is missing additional input information");

  //            auto func = hopper_input.ptr_c ?
  //            kernels::cutlass_kernels::genericMoeGemmKernelLauncherHopper<T, WeightType,
  //                            cutlass::arch::Sm90, EpilogueTag, true>
  //                                           :
  //                                           kernels::cutlass_kernels::genericMoeGemmKernelLauncherHopper<T,
  //                                           WeightType,
  //                                               cutlass::arch::Sm90, EpilogueTag, false>;
  // TODO Re-enable bias when CUTLASS supports it

  if constexpr (Arch::kMinComputeCapability < 90) {
    ORT_THROW("Invalid architecture instantiated");
  }
#ifndef COMPILE_HOPPER_TMA_GROUPED_GEMMS
  else if constexpr (Arch::kMinComputeCapability >= 90 && Arch::kMinComputeCapability < 100) {
    ORT_THROW("Please recompile with support for hopper by passing 90-real as an arch to build_wheel.py.");
  }
#endif
#ifndef COMPILE_BLACKWELL_TMA_GROUPED_GEMMS
  else if constexpr (Arch::kMinComputeCapability >= 100 && Arch::kMinComputeCapability < 120) {
    ORT_THROW("Please recompile with support for blackwell by passing 100-real as an arch to build_wheel.py.");
  }
#endif
#ifndef COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
  else if constexpr (Arch::kMinComputeCapability >= 120) {
    ORT_THROW("Please recompile with support for blackwell by passing 120-real as an arch to build_wheel.py.");
  }
#endif
  else {
    auto getFunc = [&]() {
      if constexpr (std::is_same_v<T, __nv_fp8_e4m3> && std::is_same_v<WeightType, __nv_fp4_e2m1>) {
        ORT_ENFORCE(hopper_input.fpX_block_scaling_type == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX,
                    "MXFPX is the only supported scaling type for WFP4AFP8");
        return &kernels::cutlass_kernels::tma_warp_specialized_generic_moe_gemm_kernelLauncher<Arch, T,
                                                                                               WeightType, OutputType, EpilogueTag, FUSION, TileShape, ClusterShape, true, false>;
      } else {
        ORT_ENFORCE(hopper_input.fpX_block_scaling_type != TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX,
                    "MXFPX is not supported for the selected weight combination");
        return &kernels::cutlass_kernels::tma_warp_specialized_generic_moe_gemm_kernelLauncher<Arch, T,
                                                                                               WeightType, OutputType, EpilogueTag, FUSION, TileShape, ClusterShape, false, false>;
      }
    };
    getFunc()(hopper_input, num_experts, multi_processor_count, stream, occupancy, workspace_size);
  }
}

template <typename ClusterTileShape, typename ClusterShape, typename DataType, typename WeightType>
constexpr bool are_tile_shapes_supported_sm100() {
  using namespace cute;
  using CtaShape = decltype(shape_div(ClusterTileShape{}, ClusterShape{}));
  // This is the epilogue shape. The MMA shape will be twice this for 2SM
  constexpr auto TileM = size<0>(CtaShape{});
  constexpr auto TileN = size<1>(CtaShape{});

  if constexpr (TileM != 64 && TileM != 128) {
    return false;
  }

#ifdef ENABLE_FP4
  if constexpr (std::is_same_v<DataType, __nv_fp4_e2m1> || std::is_same_v<WeightType, __nv_fp4_e2m1>) {
    // if (TileN % 64 != 0 || TileN < 128)
    // {
    //     return false;
    // }
    if ((TileN != 64 && TileN != 128 && TileN != 256) || TileM != 128) {
      return false;
    }
  }
#endif

  if constexpr (std::is_same_v<DataType, __nv_fp8_e4m3>) {
    if constexpr ((TileN == 16 || TileN == 8) && cute::size<0>(ClusterShape{}) == 1 && cute::size<1>(ClusterShape{}) == 1) {
      return true;
    }
  }

  if constexpr (TileN % 32 != 0 || TileN < 32 || TileN > 256) {
    return false;
  }

  if constexpr (cute::size<0>(ClusterShape{}) % 2 == 0 && TileN % 64 != 0) {
    return false;
  }

  return true;
}

template <typename ClusterTileShape, typename ClusterShape, typename DataType>
constexpr bool are_tile_shapes_supported_sm120() {
  using namespace cute;
  if constexpr (cute::size<0>(ClusterShape{}) != 1 || cute::size<1>(ClusterShape{}) != 1 || cute::size<2>(ClusterShape{}) != 1) {
    return false;
  }
  using CtaShape = decltype(shape_div(ClusterTileShape{}, ClusterShape{}));
  // This is the epilogue shape. The MMA shape will be twice this for 2SM
  constexpr auto TileM = size<0>(CtaShape{});
  constexpr auto TileN = size<1>(CtaShape{});
  constexpr auto TileK = size<2>(CtaShape{});

  return (TileM == 128 && TileN == 128 && TileK == 128) || (TileM == 128 && TileN == 128 && TileK == 256) || (TileM == 128 && TileN == 256 && TileK == 128) || (TileM == 256 && TileN == 128 && TileK == 128);
}

/*
    1x1x1 cluster shape is are supported for any tile shape.

    2x1x1 cluster shape is only supported for when the M tile is at least 128.

    1x2x1 cluster shape is only supported when the N tile is at least 128.

    2x2x1 cluster shape is only supported when both the M and N tiles are at least 128.

    We make the above restrictions are to improve compilation speed in TRT-LLM by pruning kernels
    that may not be very useful in practice.
 */
template <typename Arch, typename CTAShape, typename ClusterShape, typename DataType, typename WeightType>
constexpr bool are_tile_shapes_supported() {
  if constexpr (Arch::kMinComputeCapability >= 100 && Arch::kMinComputeCapability < 120) {
    return are_tile_shapes_supported_sm100<CTAShape, ClusterShape, DataType, WeightType>();
  } else if constexpr (Arch::kMinComputeCapability == 120 || Arch::kMinComputeCapability == 121) {
    return are_tile_shapes_supported_sm120<CTAShape, ClusterShape, DataType>();
  }

  using namespace cute;
  [[maybe_unused]] constexpr int cta_m = get<0>(CTAShape{});
  [[maybe_unused]] constexpr int cta_n = get<1>(CTAShape{});
  constexpr int cga_m = get<0>(ClusterShape{});
  constexpr int cga_n = get<1>(ClusterShape{});

  if constexpr (cga_m == _1{} && cga_n == _1{}) {
    return true;
  } else if constexpr (cga_m == _2{} && cga_n == _1{} && cta_m >= _128{}) {
    return true;
  } else if constexpr (cga_m == _1{} && cga_n == _2{} && cta_n >= _128{}) {
    return true;
  } else if constexpr (cga_m == _2{} && cga_n == _2{} && cta_m >= _128{} && cta_n >= _128{}) {
    return true;
  } else {
    return false;
  }
}

template <typename Arch, typename T, typename WeightType, typename OutputType, typename EpilogueTag,
          EpilogueFusion FUSION, typename TileShape>
void dispatchMoeGemmSelectClusterShapeTmaWarpSpecialized(TmaWarpSpecializedGroupedGemmInput hopper_input,
                                                         int num_experts, cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
                                                         int* occupancy, size_t* workspace_size) {
  using namespace cute;
  switch (gemm_config.cluster_shape) {
#define SHAPE_CASE(M, N, K)                                                                                    \
  case cutlass_extensions::ClusterShape::ClusterShape_##M##x##N##x##K: {                                       \
    using ClusterShape = Shape<_##M, _##N, _##K>;                                                              \
    if constexpr (are_tile_shapes_supported<Arch, TileShape, ClusterShape, T, WeightType>()) {                 \
      dispatchMoeGemmSelectBiasTmaWarpSpecialized<Arch, T, WeightType, OutputType, EpilogueTag, FUSION,        \
                                                  TileShape, ClusterShape>(                                    \
          hopper_input, num_experts, multi_processor_count, stream, occupancy, workspace_size);                \
      break;                                                                                                   \
    } else {                                                                                                   \
      ORT_THROW(                                                                                               \
          "%s\nUnsupported tile (%d, %d, %d) and cluster (%d, %d, %d) shape combination for arch %d.\nConfig " \
          "was %s",                                                                                            \
          __PRETTY_FUNCTION__, (int)cute::get<0>(TileShape{}), (int)cute::get<1>(TileShape{}),                 \
          (int)cute::get<2>(TileShape{}), M, N, K, (int)Arch::kMinComputeCapability,                           \
          gemm_config.toString().c_str());                                                                     \
    }                                                                                                          \
  }

    SHAPE_CASE(1, 1, 1)
    SHAPE_CASE(1, 2, 1)

    SHAPE_CASE(2, 1, 1)
    SHAPE_CASE(2, 2, 1)

#undef SHAPE_CASE
    default:
      ORT_THROW("Unsupported config %d for MoE gemm.", (int)gemm_config.cluster_shape);
  }
}

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag, EpilogueFusion FUSION>
void dispatchMoeGemmSelectTileShapeTmaWarpSpecialized(TmaWarpSpecializedGroupedGemmInput hopper_input, int num_experts,
                                                      cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream, int* occupancy,
                                                      size_t* workspace_size) {
  using namespace cute;

#define SHAPE_CASE(SMVERSION, M, N, K)                                                                                                      \
  case cutlass_extensions::CutlassTileConfigSM##SMVERSION::CtaShape##M##x##N##x##K##B: {                                                    \
    constexpr int KtileBytes = (K * 8) / cutlass::sizeof_bits<typename kernels::cutlass_kernels::CudaToCutlassTypeAdapter<T>::type>::value; \
    using KTileDim = Int<KtileBytes>;                                                                                                       \
    using TileShape = Shape<_##M, _##N, KTileDim>;                                                                                          \
    dispatchMoeGemmSelectClusterShapeTmaWarpSpecialized<cutlass::arch::Sm##SMVERSION, T, WeightType, OutputType,                            \
                                                        EpilogueTag, FUSION, TileShape>(                                                    \
        hopper_input, num_experts, gemm_config, multi_processor_count, stream, occupancy, workspace_size);                                  \
    break;                                                                                                                                  \
  }
#define DEFAULT_CASE(SMVERSION)                                                                   \
  case cutlass_extensions::CutlassTileConfigSM##SMVERSION::Undefined:                             \
    ORT_THROW("GEMM config undefined.");                                                          \
    break;                                                                                        \
  case cutlass_extensions::CutlassTileConfigSM##SMVERSION::ChooseWithHeuristic:                   \
    ORT_THROW("GEMM config should have already been set by heuristic.");                          \
    break;                                                                                        \
  default:                                                                                        \
    ORT_THROW("Unsupported config %d for MoE gemm.", (int)gemm_config.tile_config_sm##SMVERSION); \
    break;

  if (gemm_config.sm_version == 90) {
    if constexpr (kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType, EpilogueTag, FUSION>()) {
      switch (gemm_config.tile_config_sm90) {
        SHAPE_CASE(90, 128, 16, 128)
        SHAPE_CASE(90, 128, 32, 128)
        SHAPE_CASE(90, 128, 64, 128)
        SHAPE_CASE(90, 128, 128, 128)
        SHAPE_CASE(90, 128, 256, 128)
        SHAPE_CASE(90, 256, 128, 128)
        DEFAULT_CASE(90)
      }
    } else {
      ORT_THROW("Unsupported SM90 configuration requested");
    }
  } else if (gemm_config.sm_version >= 100 && gemm_config.sm_version < 120) {
    if constexpr (kernels::cutlass_kernels::isValidBlackwellMOESpecialisation<T, WeightType, EpilogueTag, FUSION>()) {
      switch (gemm_config.tile_config_sm100) {
        SHAPE_CASE(100, 64, 64, 128)
        SHAPE_CASE(100, 64, 128, 128)
        SHAPE_CASE(100, 64, 256, 128)

        SHAPE_CASE(100, 128, 16, 128)
        SHAPE_CASE(100, 128, 32, 128)
        SHAPE_CASE(100, 128, 64, 128)
        SHAPE_CASE(100, 128, 128, 128)
        SHAPE_CASE(100, 128, 256, 128)

        SHAPE_CASE(100, 256, 64, 128)
        SHAPE_CASE(100, 256, 128, 128)
        SHAPE_CASE(100, 256, 256, 128)

        // SHAPE_CASE(100, 128, 128, 64)
        // SHAPE_CASE(100, 128, 256, 64)
        // SHAPE_CASE(100, 256, 256, 64)
        DEFAULT_CASE(100)
      }
    } else {
      ORT_THROW("Unsupported SM100 configuration requested");
    }
  } else if (gemm_config.sm_version == 120 || gemm_config.sm_version == 121) {
    ORT_LLM_LOG_TRACE(onnxruntime::MakeString("At ", __PRETTY_FUNCTION__, "SM120 config=", gemm_config.tile_config_sm120));
    if constexpr (kernels::cutlass_kernels::isValidSM120MOESpecialisation<T, WeightType, EpilogueTag, FUSION>()) {
      switch (gemm_config.tile_config_sm120) {
        SHAPE_CASE(120, 128, 128, 64)
        SHAPE_CASE(120, 128, 128, 128)
        SHAPE_CASE(120, 128, 256, 64)
        SHAPE_CASE(120, 256, 128, 64)
        DEFAULT_CASE(120)
      }
    }
  }
#undef SHAPE_CASE
}

template <typename T, typename WeightType, typename OutputType, EpilogueFusion FUSION>
size_t calcMaxWorkspaceSizeTmaWarpSpecialized(int num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                                              int multi_processor_count, TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType fpX_block_scaling_type) {
  size_t count = 0;
  TmaWarpSpecializedGroupedGemmInput input{};
  input.fpX_block_scaling_type = fpX_block_scaling_type;
  // Most of the values are ignored for WS size calculation. We reuse the function to reduce the template bloat
  dispatchMoeGemmSelectTileShapeTmaWarpSpecialized<T, WeightType, OutputType, cutlass_extensions::EpilogueOpDefault,
                                                   FUSION>(input, num_experts, gemm_config, multi_processor_count, cudaStream_t{0}, nullptr, &count);
  return count;
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
