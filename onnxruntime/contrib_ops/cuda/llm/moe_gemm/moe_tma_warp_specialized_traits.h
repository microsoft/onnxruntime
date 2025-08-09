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

#include "cutlass/arch/mma_sm90.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

namespace onnxruntime::llm::kernels::cutlass_kernels {

// Blackwell arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidSM120MOESpecialisation() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)  // TODO Is there a better choice
  return cutlass::platform::is_same<T, __nv_fp4_e2m1>::value && cutlass::platform::is_same<T, WeightType>::value && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value && Fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
#else
  return false;  // CUTLASS_ARCH_MMA_SM100_SUPPORTED is set when Blackwell kernels are enabled
#endif
}

template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidBlackwellMOESpecialisation() {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)  // TODO Is there a better choice
  return (cutlass::platform::is_same<T, WeightType>::value || (cutlass::platform::is_same<T, __nv_fp8_e4m3>::value && cutlass::platform::is_same<WeightType, __nv_fp4_e2m1>::value)) && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value && Fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
#else
  return false;  // CUTLASS_ARCH_MMA_SM100_SUPPORTED is set when Blackwell kernels are enabled
#endif
}

// Hopper arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidHopperMOESpecialisation() {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  return (cutlass::platform::is_same<T, WeightType>::value || (cutlass::platform::is_same<cutlass::uint4b_t, WeightType>::value && cutlass::platform::is_same<T, __nv_fp8_e4m3>::value))
#ifdef ENABLE_FP4
         && !cutlass::platform::is_same<T, __nv_fp4_e2m1>::value
#endif
         && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#else
  return false;  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED is set when Hopper kernels are enabled
#endif
}

template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidTmaWarpSpecializedMOESpecialisation() {
  // Check at least one of the implementations are valid
  return isValidBlackwellMOESpecialisation<T, WeightType, EpilogueTag, Fusion>() || isValidHopperMOESpecialisation<T, WeightType, EpilogueTag, Fusion>();
}

// Hopper arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidAmpereMOESpecialisation() {
#ifdef ENABLE_FP4
  return !std::is_same_v<T, __nv_fp4_e2m1> && !std::is_same_v<WeightType, __nv_fp4_e2m1>;
#else
  return true;  // Default to true
#endif
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
