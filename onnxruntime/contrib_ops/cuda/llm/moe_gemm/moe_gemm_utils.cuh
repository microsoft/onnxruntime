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

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cuda/llm/kernels/quantization.cuh"
#include <cutlass/numeric_conversion.h>

namespace onnxruntime::llm::kernels::cutlass_kernels {

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices, int64_t const arr_length, T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

template <class T>
using sizeof_bits = cutlass::sizeof_bits<typename cutlass_kernels::CudaToCutlassTypeAdapter<std::remove_cv_t<T>>::type>;

// Function to safely offset an pointer that may contain sub-byte types (FP4/INT4)
template <class T>
__host__ __device__ constexpr T* safe_inc_ptr(T* ptr, size_t offset) {
  constexpr int adjustment = (cutlass::sizeof_bits<typename CudaToCutlassTypeAdapter<std::remove_cv_t<T>>::type>::value < 8) ? (8 / cutlass::sizeof_bits<typename CudaToCutlassTypeAdapter<std::remove_cv_t<T>>::type>::value) : 1;
  assert(offset % adjustment == 0 && "Attempt to offset index to sub-byte");
  return ptr + offset / adjustment;
}

__host__ __device__ constexpr int64_t getOffsetWeightSF(int64_t expert_id, int64_t gemm_n, int64_t gemm_k,
                                                        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment, int64_t block_size) {
    int64_t padded_gemm_n = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_n, min_n_dim_alignment);
    int64_t padded_gemm_k = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    return expert_id * padded_gemm_n * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

__host__ __device__ constexpr int64_t getOffsetActivationSF(int64_t expert_id, int64_t token_offset, int64_t gemm_k,
                                                            TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment, int64_t block_size) {
    // This formulation ensures that:
    // `sf_offset[i + 1] - sf_offset[i] >= padded(token_offset[i + 1] - token_offset[i])`
    // is true for all possible token distributions.
    int64_t padded_sf_start_offset = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
        token_offset + expert_id * (min_n_dim_alignment - 1), min_n_dim_alignment);
    int64_t padded_gemm_k = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    assert(padded_gemm_k % block_size == 0);
    return padded_sf_start_offset * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

template <class GemmOutputType, class QuantizedType, class ComputeElem, int VecSize>
__device__ auto quantizePackedFPXValue(ComputeElem& post_act_val, float global_scale_val,
                                       int64_t num_tokens_before_expert, int64_t expert_id, int64_t token_id, int64_t elem_idx, int64_t num_cols,
                                       TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
                                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  constexpr bool is_fp8 = std::is_same_v<QuantizedType, __nv_fp8_e4m3>;
  static constexpr int NumThreadsPerSF = VecSize / CVT_FP4_ELTS_PER_THREAD;
  // Quantize the input to FP4
  static_assert(std::is_same_v<GemmOutputType, __nv_bfloat16> || std::is_same_v<GemmOutputType, half>);
  static_assert(ComputeElem::kElements == CVT_FP4_ELTS_PER_THREAD);
  PackedVec<GemmOutputType> packed_vec{};
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    packed_vec.elts[i].x = static_cast<GemmOutputType>(post_act_val[i * 2 + 0]);
    packed_vec.elts[i].y = static_cast<GemmOutputType>(post_act_val[i * 2 + 1]);
  }

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols, scaling_type);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
  auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF, VecSize>(
      std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx, std::nullopt /* numRows */,
      num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);

  // Do the conversion and set the output and scaling factor
  auto func = [&]() {
    if constexpr (is_fp8) {
      return [](PackedVec<GemmOutputType>& vec, float /* ignored */, uint8_t* SFout) -> uint64_t {
        static_assert(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize == VecSize);
        return cvt_warp_fp16_to_mxfp8<GemmOutputType, VecSize>(vec, SFout);
      };
    } else {
      return (scaling_type == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4)
                 ? &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, false>
                 : &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, true>;
    }
  }();

  return func(packed_vec, global_scale_val, sf_out);
}

template <int VecSize, int ElementsPerThread>
__device__ void writeSF(int64_t num_tokens_before_expert, int64_t expert_id, int64_t source_token_id, int64_t token_id,
                        int64_t elem_idx, int64_t num_cols, TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
                        TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf) {
  static constexpr int NumThreadsPerSF = VecSize / ElementsPerThread;

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert, num_cols,
                                                           (VecSize == TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize)
                                                               ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                                                               : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this expert
  auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF, VecSize>(
      std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx, std::nullopt /* numRows */,
      num_cols, act_sf_expert, FP4QuantizationSFLayout::SWIZZLED);
  if (sf_out) {
    if (input_sf) {
      auto const sf_in = cvt_quant_to_fp4_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF,
                                                            VecSize>(std::nullopt /* batchIdx */, source_token_id, elem_idx, std::nullopt /* numRows */,
                                                                     num_cols, const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
                                                                     FP4QuantizationSFLayout::SWIZZLED);
      *sf_out = *sf_in;
    } else {
      *sf_out = 0x00;
    }
  }
}

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
