// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/framework/int4.h"
#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
#include "core/framework/float4.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Identifies the FP8/FP4 data types that GatherBlockQuantized dequantizes symmetrically
// (output = float(data) * scale), as opposed to the integer block-quantized types
// (uint8/int4/uint4) which are dequantized as (code - zero_point) * scale.
template <typename T1>
struct IsFpQuantized : std::false_type {};

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
struct IsFpQuantized<Float8E4M3FN> : std::true_type {};
template <>
struct IsFpQuantized<Float8E4M3FNUZ> : std::true_type {};
template <>
struct IsFpQuantized<Float8E5M2> : std::true_type {};
template <>
struct IsFpQuantized<Float8E5M2FNUZ> : std::true_type {};
#endif  // !defined(DISABLE_FLOAT4_TYPES)
#if !defined(DISABLE_FLOAT4_TYPES)
template <>
struct IsFpQuantized<Float4E2M1x2> : std::true_type {};
#endif  // !defined(DISABLE_FLOAT4_TYPES)

template <typename T1>
inline constexpr bool IsFpQuantizedV = IsFpQuantized<T1>::value;

struct GatherBlockQuantizedParam {
  cudaStream_t stream;
  int64_t after_gather_dim;
  int64_t gather_axis_dim;
  int64_t ind_dim;
  int64_t bits;
  // For FP8/FP4 data this is the *effective* block size: block_size_ (attribute) if nonzero,
  // otherwise the full quantize_axis dimension (block_size == 0 means "one scale per row").
  int64_t block_size;
  int64_t gather_axis;
  int64_t N;
  // Total number of elements in `scales`. When this is 1, every output element is dequantized
  // with the single (broadcast) scale value, regardless of block_id. Only used for FP8/FP4 data;
  // partial broadcasting (e.g. a single scale per row and nothing else) is not supported on CUDA.
  int64_t scale_size;
};

template <typename T1, typename T2, typename Tind>
void LaunchGatherBlockQuantizedKernel(const T1* data,
                                      const Tind* indices,
                                      const T2* scales,
                                      const T1* zero_points,
                                      T2* output,
                                      GatherBlockQuantizedParam param);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
