// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ROCm GatherBlockQuantized CUDA kernels.
//
// Pure HIP, no arch-specific intrinsics.  Correct on gfx900 and all newer targets.

#include "contrib_ops/rocm/quantization/gather_block_quantized.cuh"
#include <hip/hip_fp16.h>
#include "core/framework/int4.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

// ---------------------------------------------------------------------------
// Nibble extraction (signed and unsigned)
// ---------------------------------------------------------------------------
template <typename T1>
__device__ __forceinline__ long long ExtractSignedVal(
    const T1* data, long long idx, long long bits) {
  const long long elems_per_byte = 8LL / bits;
  const long long byte_idx   = idx / elems_per_byte;
  const long long bit_offset = (idx % elems_per_byte) * bits;
  const unsigned char byte =
      reinterpret_cast<const unsigned char*>(data)[byte_idx];
  const unsigned long long mask = (1ULL << bits) - 1ULL;
  long long val = static_cast<long long>((byte >> bit_offset) & mask);
  if (val & (1LL << (bits - 1))) {
    val |= ~0LL << bits;  // sign-extend
  }
  return val;
}

template <typename T1>
__device__ __forceinline__ long long ExtractUnsignedVal(
    const T1* data, long long idx, long long bits) {
  const long long elems_per_byte = 8LL / bits;
  const long long byte_idx   = idx / elems_per_byte;
  const long long bit_offset = (idx % elems_per_byte) * bits;
  const unsigned char byte =
      reinterpret_cast<const unsigned char*>(data)[byte_idx];
  const unsigned long long mask = (1ULL << bits) - 1ULL;
  return static_cast<long long>((byte >> bit_offset) & mask);
}

// ---------------------------------------------------------------------------
// Core kernel
// ---------------------------------------------------------------------------
template <typename T1, typename T2, typename Tind, bool IsSigned>
__global__ void GatherBlockQuantizedKernel(
    const T1* __restrict__ data,
    const Tind* __restrict__ indices,
    const T2* __restrict__ scales,
    const T1* __restrict__ zero_points,
    T2* __restrict__ output,
    long long after_gather_dim,
    long long gather_axis_dim,
    long long ind_dim,
    long long bits,
    long long block_size,
    long long N) {
  const long long out_idx =
      static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (out_idx >= N) return;

  const long long idx_before =
      out_idx / (after_gather_dim * ind_dim);
  const long long tmp = out_idx % (after_gather_dim * ind_dim);
  const long long idx_pos   = tmp / after_gather_dim;
  const long long idx_after = tmp % after_gather_dim;

  const long long idx_at_g =
      static_cast<long long>(indices[idx_pos]);

  const long long in_idx =
      idx_before * gather_axis_dim * after_gather_dim +
      idx_at_g * after_gather_dim +
      idx_after;

  const long long block_id = in_idx / block_size;

  long long zp = 0;
  if (zero_points) {
    if constexpr (IsSigned) {
      zp = ExtractSignedVal(zero_points, block_id, bits);
    } else {
      zp = ExtractUnsignedVal(zero_points, block_id, bits);
    }
  }

  long long weight = 0;
  if constexpr (IsSigned) {
    weight = ExtractSignedVal(data, in_idx, bits);
  } else {
    weight = ExtractUnsignedVal(data, in_idx, bits);
  }

  output[out_idx] =
      static_cast<T2>(weight - zp) * scales[block_id];
}

// ---------------------------------------------------------------------------
// Typed launch wrapper
// ---------------------------------------------------------------------------
template <typename T1, typename T2, typename Tind>
void LaunchGatherBlockQuantizedKernel(
    const T1* data,
    const Tind* indices,
    const T2* scales,
    const T1* zero_points,
    T2* output,
    GatherBlockQuantizedParam param) {
  constexpr int kBlock = 256;
  const int grid =
      static_cast<int>((param.N + kBlock - 1) / kBlock);

  constexpr bool is_signed = std::is_same_v<T1, Int4x2>;

  GatherBlockQuantizedKernel<T1, T2, Tind, is_signed>
      <<<grid, kBlock, 0, param.stream>>>(
          data, indices, scales, zero_points, output,
          param.after_gather_dim,
          param.gather_axis_dim,
          param.ind_dim,
          param.bits,
          param.block_size,
          param.N);
}

// ---------------------------------------------------------------------------
// Explicit instantiations  (must match the REGISTER_GBQ set in .cc)
// ---------------------------------------------------------------------------
#define INSTANTIATE_GBQ(T1, T2, Tind)                               \
  template void LaunchGatherBlockQuantizedKernel<T1, T2, Tind>(     \
      const T1*, const Tind*, const T2*, const T1*, T2*,            \
      GatherBlockQuantizedParam);

INSTANTIATE_GBQ(uint8_t, float,  int32_t)
INSTANTIATE_GBQ(uint8_t, float,  int64_t)
INSTANTIATE_GBQ(uint8_t, __half, int32_t)
INSTANTIATE_GBQ(uint8_t, __half, int64_t)

INSTANTIATE_GBQ(UInt4x2, float,  int32_t)
INSTANTIATE_GBQ(UInt4x2, float,  int64_t)
INSTANTIATE_GBQ(UInt4x2, __half, int32_t)
INSTANTIATE_GBQ(UInt4x2, __half, int64_t)

INSTANTIATE_GBQ(Int4x2, float,  int32_t)
INSTANTIATE_GBQ(Int4x2, float,  int64_t)
INSTANTIATE_GBQ(Int4x2, __half, int32_t)
INSTANTIATE_GBQ(Int4x2, __half, int64_t)

#undef INSTANTIATE_GBQ

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
