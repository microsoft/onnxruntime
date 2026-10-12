// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dynamic_quantize_matmul_nbits.h"

#include <algorithm>
#include <cfloat>
#include <limits>
#include <type_traits>
#include <math_constants.h>

#include "core/common/safeint.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime::contrib::cuda {
namespace {
constexpr int kThreads = onnxruntime::cuda::GridDim::maxThreadsPerBlock;
constexpr int kTensorReductionMaxBlocks = 1024;
constexpr int kTensorReductionElementsPerBlock = 4096;

int Blocks(int64_t count, int work_per_block = kThreads) {
  return static_cast<int>(std::min<int64_t>(65535, count / work_per_block + (count % work_per_block != 0)));
}

__device__ float LoadScale(const void* scales, DynamicWeightScaleType type, int64_t index) {
  if (type == DynamicWeightScaleType::Float) return static_cast<const float*>(scales)[index];
  if (type == DynamicWeightScaleType::Half) return to_float(static_cast<const half*>(scales)[index]);
  return to_float(static_cast<const __nv_bfloat16*>(scales)[index]);
}

__global__ void ValidateScales(const void* scales, DynamicWeightScaleType type, int64_t count, int* invalid) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count;
       i += int64_t{gridDim.x} * blockDim.x) {
    if (!isfinite(LoadScale(scales, type, i))) atomicExch(invalid, 1);
  }
}

__global__ void DynamicSignedWeights(const uint8_t* weights, int8_t* signed_weights, int64_t count) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count / 4;
       i += int64_t{gridDim.x} * blockDim.x) {
    reinterpret_cast<uint32_t*>(signed_weights)[i] = reinterpret_cast<const uint32_t*>(weights)[i] ^ 0x80808080u;
  }
}

__device__ float ReduceMaximum(float maximum, float* shared) {
  for (int delta = 16; delta > 0; delta /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
  }
  if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = maximum;
  __syncthreads();
  if (threadIdx.x < 32) {
    maximum = threadIdx.x < blockDim.x / 32 ? shared[threadIdx.x] : 0.0f;
    for (int delta = 16; delta > 0; delta /= 2) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    }
    if (threadIdx.x == 0) shared[0] = maximum;
  }
  __syncthreads();
  return shared[0];
}

__device__ float ScaleFromMaximum(float maximum) {
  if (!isfinite(maximum)) return CUDART_NAN_F;
  if (maximum == 0) return 1.0f;
  return fmaxf(__fdiv_rn(maximum, 127.0f), __uint_as_float(1));
}

// Infinity marks either infinity or NaN; max reduction then preserves the marker.
template <typename T>
__device__ float DomainScale(const T* a, int64_t count, float* shared) {
  float maximum = 0;
  for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
    const float value = to_float(a[i]);
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : CUDART_INF_F);
  }
  return ScaleFromMaximum(ReduceMaximum(maximum, shared));
}

template <typename T>
__global__ void TensorScale(const T* a, int64_t count, float* scale) {
  __shared__ float shared[kThreads / 32];
  const float value = DomainScale(a, count, shared);
  if (threadIdx.x == 0) *scale = value;
}

template <typename T>
__global__ void TensorMaximumPartials(const T* a, int64_t count, float* partials) {
  __shared__ float shared[kThreads / 32];
  float maximum = 0.0f;
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count;
       i += int64_t{gridDim.x} * blockDim.x) {
    const float value = to_float(a[i]);
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : CUDART_INF_F);
  }
  maximum = ReduceMaximum(maximum, shared);
  if (threadIdx.x == 0) partials[blockIdx.x] = maximum;
}

__global__ void TensorScaleFromPartials(const float* partials, int count, float* scale) {
  __shared__ float shared[kThreads / 32];
  float maximum = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) maximum = fmaxf(maximum, partials[i]);
  maximum = ReduceMaximum(maximum, shared);
  if (threadIdx.x == 0) *scale = ScaleFromMaximum(maximum);
}

template <typename T>
__global__ void QuantizeRows(const T* a, int64_t rows, int64_t k, int64_t block_size,
                             const float* tensor_scale, int8_t* quantized, float* scales) {
  __shared__ float shared[kThreads / 32];
  const int64_t groups = k / block_size + (k % block_size != 0);
  for (int64_t domain = blockIdx.x; domain < rows * groups; domain += gridDim.x) {
    const int64_t row = domain / groups;
    const int64_t start = (domain % groups) * block_size;
    const int64_t count = min(block_size, k - start);
    const float scale = tensor_scale ? *tensor_scale : DomainScale(a + row * k + start, count, shared);
    if (threadIdx.x == 0 && scales) scales[domain] = scale;
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
      const int64_t index = row * k + start + i;
      const int code = isnan(scale) ? 0 : __float2int_rn(__fdiv_rn(to_float(a[index]), scale));
      quantized[index] = static_cast<int8_t>(max(-127, min(127, code)));
    }
    __syncthreads();
  }
}

constexpr int kVectorMaxChunks = 4;
constexpr int kVectorElements = 8;

// One group of G threads quantizes one domain whose values stay in registers (one global read).
// Each thread owns up to Chunks 16-byte chunks, so a domain holds at most G * Chunks * 8 elements.
// Arithmetic is identical to QuantizeRows: max is order independent and codes use the same rounding.
template <typename T, int G, int Chunks>
__global__ void QuantizeRowsVector(const T* a, int64_t domains, int64_t k, int64_t block_size, int64_t groups,
                                   const float* tensor_scale, int8_t* quantized, float* scales) {
  __shared__ float shared[kThreads / 32];
  constexpr int kDomainsPerBlock = kThreads / G;
  const int64_t domain = int64_t{blockIdx.x} * kDomainsPerBlock + threadIdx.x / G;
  const bool active = domain < domains;
  const int lane = threadIdx.x % G;
  const int64_t row = active ? domain / groups : 0;
  const int64_t start = active ? (domain % groups) * block_size : 0;
  const int64_t count = active ? min(block_size, k - start) : 0;
  const int64_t base = row * k + start;
  const int chunks = static_cast<int>(count / kVectorElements);

  uint4 raw[Chunks];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < Chunks; ++i) {
    const int c = lane + i * G;
    if (c < chunks) {
      raw[i] = reinterpret_cast<const uint4*>(a + base)[c];
      if (!tensor_scale) {
        const T* e = reinterpret_cast<const T*>(&raw[i]);
#pragma unroll
        for (int j = 0; j < kVectorElements; ++j) {
          const float value = to_float(e[j]);
          maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : CUDART_INF_F);
        }
      }
    }
  }

  float scale;
  if (tensor_scale) {
    scale = *tensor_scale;
  } else {
    for (int delta = 16; delta > 0; delta /= 2) {
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    }
    if constexpr (G > 32) {
      if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = maximum;
      __syncthreads();
      const float* group_shared = shared + (threadIdx.x / G) * (G / 32);
      maximum = group_shared[0];
#pragma unroll
      for (int w = 1; w < G / 32; ++w) maximum = fmaxf(maximum, group_shared[w]);
    }
    scale = ScaleFromMaximum(maximum);
  }
  if (!active) return;
  if (lane == 0 && scales) scales[domain] = scale;

#pragma unroll
  for (int i = 0; i < Chunks; ++i) {
    const int c = lane + i * G;
    if (c < chunks) {
      const T* e = reinterpret_cast<const T*>(&raw[i]);
      uint32_t packed[2];
#pragma unroll
      for (int j = 0; j < kVectorElements; j += 4) {
        uint32_t word = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int code = isnan(scale) ? 0 : __float2int_rn(__fdiv_rn(to_float(e[j + q]), scale));
          word |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(max(-127, min(127, code))))) << (8 * q);
        }
        packed[j / 4] = word;
      }
      reinterpret_cast<uint2*>(quantized + base)[c] = make_uint2(packed[0], packed[1]);
    }
  }
}

__device__ __noinline__ float ScaleProductSlow(int32_t acc, float a, float b) {
  return __double2float_rn(__dmul_rn(__dmul_rn(static_cast<double>(__int2float_rn(acc)), static_cast<double>(a)),
                                     static_cast<double>(b)));
}

template <bool NormalActivationProduct = false>
__device__ __forceinline__ float ScaleProduct(int32_t acc, float a, float b) {
  if (isnan(a)) return CUDART_NAN_F;
  const float t = __fmul_rn(__int2float_rn(acc), a);
  // Finite FP16 domain scales are in [2^-24 / 127, 65504 / 127].
  // A nonzero INT32 accumulator times such a scale is always normal FP32.
  if constexpr (NormalActivationProduct) return __fmul_rn(t, b);
  if (acc == 0 || (isfinite(t) && fabsf(t) >= FLT_MIN)) return __fmul_rn(t, b);
  return ScaleProductSlow(acc, a, b);
}

template <typename T>
__device__ __forceinline__ float DecodeScaleProduct(int32_t acc, float a, float b, bool normal_activation_scale) {
  if constexpr (std::is_same_v<T, half>) {
    return ScaleProduct<true>(acc, a, b);
  }
  if (normal_activation_scale) {
    return __fmul_rn(__fmul_rn(__int2float_rn(acc), a), b);
  }
  return ScaleProduct<false>(acc, a, b);
}

template <typename T>
__device__ __forceinline__ float LoadTypedScale(const void* scales, int64_t index) {
  if constexpr (std::is_same_v<T, float>) {
    return static_cast<const float*>(scales)[index];
  }
  return to_float(static_cast<const T*>(scales)[index]);
}

template <int Bits>
__device__ __forceinline__ int DecodeWeight(const uint8_t* blob, int64_t j) {
  constexpr int pack = 8 / Bits;
  return ((blob[j / pack] >> (Bits * (j % pack))) & ((1 << Bits) - 1)) - (1 << (Bits - 1));
}

// Four consecutive centered codes. The caller guarantees they belong to one blob.
template <int Bits>
__device__ __forceinline__ uint32_t DecodeWeight4(const uint8_t* blob, int64_t j) {
  if constexpr (Bits == 8) {
    if (j % 4 == 0) return *reinterpret_cast<const uint32_t*>(blob + j) ^ 0x80808080u;
  } else if (j % 4 == 0) {
    uint32_t codes;
    if constexpr (Bits == 4) {
      const uint32_t packed = *reinterpret_cast<const uint16_t*>(blob + j / 2);
      const uint32_t spaced = __byte_perm(packed, 0, 0x4140);
      codes = (spaced & 0x000f000fu) | ((spaced & 0x00f000f0u) << 4);
    } else {
      const uint32_t packed = blob[j / 4];
      codes = (packed & 0x03u) | ((packed & 0x0cu) << 6) |
              ((packed & 0x30u) << 12) | ((packed & 0xc0u) << 18);
    }
    // Set each byte's high bit to prevent subtraction borrowing between codes.
    constexpr uint32_t midpoint = Bits == 4 ? 0x08080808u : 0x02020202u;
    return ((codes | 0x80808080u) - midpoint) ^ 0x80808080u;
  }
  uint32_t codes = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    codes |= static_cast<uint32_t>(static_cast<uint8_t>(DecodeWeight<Bits>(blob, j + i))) << (8 * i);
  }
  return codes;
}

__device__ __forceinline__ uint32_t DecodePrepackedInt4(const uint8_t* blob, int quarter, bool upper) {
  const int base = (quarter % 2) * 16 + quarter / 2 + (upper ? 2 : 0);
  const int word = base / 8;
  const int shift = (base % 8) * 4;
  const auto* words = reinterpret_cast<const uint32_t*>(blob);
  const uint32_t first = words[word] >> shift;
  const uint32_t second = words[word + 1] >> shift;
  const uint32_t codes = (first & 0xfu) | (((first >> 16) & 0xfu) << 8) |
                         ((second & 0xfu) << 16) | (((second >> 16) & 0xfu) << 24);
  return ((codes | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

// Four consecutive centered codes of one B fragment register in the SM80 fpA_intB layout. `column` points at the
// column's words inside one 64-element K tile; `tile_step` selects the 32-wide half of that tile.
// The tile layout is the row permutation, column-major pair interleave and in-word pair interleave of
// preprocess_weights_for_mixed_gemm_cuda.
template <int Bits>
__device__ __forceinline__ uint32_t DecodePrepacked(const uint32_t* column, int tile_step, int quarter, bool upper) {
  static_assert(Bits == 2 || Bits == 4 || Bits == 8);
  if constexpr (Bits == 4) {
    return DecodePrepackedInt4(reinterpret_cast<const uint8_t*>(column + tile_step * 4), quarter, upper);
  } else if constexpr (Bits == 8) {
    const int word = (tile_step * 2 + (upper ? 1 : 0)) * 4 + (quarter % 2) * 2;
    const uint32_t selector = 0x6420u + 0x1111u * (quarter / 2);
    return __byte_perm(column[word], column[word + 1], selector) ^ 0x80808080u;
  } else {
    const int shift = 2 * (tile_step * 4 + (upper ? 2 : 0) + quarter / 2);
    const uint32_t first = (column[(quarter % 2) * 2] >> shift) & 0x00030003u;
    const uint32_t second = (column[(quarter % 2) * 2 + 1] >> shift) & 0x00030003u;
    const uint32_t codes = __byte_perm(first, second, 0x6420u);
    return ((codes | 0x80808080u) - 0x02020202u) ^ 0x80808080u;
  }
}

// Lossless signed-INT8 expansion of centered codes for the Hopper grouped-scale GEMM.
template <int Bits>
__global__ void DynamicExpandWeights(const uint8_t* weights, int8_t* expanded, int64_t count) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count / 4;
       i += int64_t{gridDim.x} * blockDim.x) {
    reinterpret_cast<uint32_t*>(expanded)[i] = DecodeWeight4<Bits>(weights, i * 4);
  }
}

__device__ __forceinline__ int32_t Dot4(uint32_t a, uint32_t b, int32_t acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(static_cast<int>(a), static_cast<int>(b), acc);
#else
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc += static_cast<int8_t>(a >> (8 * i)) * static_cast<int8_t>(b >> (8 * i));
  }
  return acc;
#endif
}

template <typename T, int Bits>
__global__ void IntegerMatMul(const int8_t* a, const float* a_scales, const uint8_t* b,
                              const void* b_scales, DynamicWeightScaleType scale_type,
                              const T* bias, T* y, bool per_tensor, int64_t rows, int64_t n, int64_t k,
                              int64_t block_size, int64_t activation_block_size) {
  constexpr int warp_size = 32;
  const int lane = threadIdx.x % warp_size;
  const int64_t groups = k / block_size + (k % block_size != 0);
  const int64_t activation_groups = per_tensor ? 0 : k / activation_block_size + (k % activation_block_size != 0);
  constexpr int pack = 8 / Bits;
  const int64_t blob_size = block_size / pack;
  for (int64_t output = (int64_t{blockIdx.x} * blockDim.x + threadIdx.x) / warp_size;
       output < rows * n; output += int64_t{gridDim.x} * (blockDim.x / warp_size)) {
    const int64_t row = output / n;
    const int64_t column = output % n;
    float total = 0;
    for (int64_t start = 0; start < k;) {
      const int64_t weight_group = start / block_size;
      const int64_t activation_group = start / activation_block_size;
      const int64_t length = min(k - start, min(block_size - start % block_size,
                                                activation_block_size - start % activation_block_size));
      int32_t acc = 0;
      const uint8_t* blob = b + (column * groups + weight_group) * blob_size;
      for (int64_t offset = lane * 4; offset < length; offset += warp_size * 4) {
        const int64_t pos = start + offset;
        const int64_t j = pos % block_size;
        if (offset + 4 <= length) {
          const int8_t* activation = a + row * k + pos;
          uint32_t codes = 0;
          if ((row * k + pos) % 4 == 0) {
            codes = *reinterpret_cast<const uint32_t*>(activation);
          } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              codes |= static_cast<uint32_t>(static_cast<uint8_t>(activation[i])) << (8 * i);
            }
          }
          acc = Dot4(codes, DecodeWeight4<Bits>(blob, j), acc);
        } else {
          for (int i = 0; i < length - offset; ++i) {
            acc += static_cast<int32_t>(a[row * k + pos + i]) * DecodeWeight<Bits>(blob, j + i);
          }
        }
      }
      for (int delta = 16; delta > 0; delta /= 2) {
        const int32_t other = __shfl_down_sync(0xffffffff, acc, delta);
        if (lane < delta) acc += other;
      }
      if (lane == 0) {
        const float term = ScaleProduct<std::is_same_v<T, half>>(acc, a_scales[row * activation_groups + activation_group],
                                                                 LoadScale(b_scales, scale_type, column * groups + weight_group));
        total = start == 0 ? term : __fadd_rn(total, term);
      }
      start += length;
    }
    if (lane == 0) y[output] = from_float<T>(__fadd_rn(total, bias ? to_float(bias[column]) : 0.0f));
  }
}

constexpr int kNativeM = 32;
constexpr int kNativeN = 64;
constexpr int kNativeK = 64;
constexpr int kNativeThreads = 128;
// An odd word stride distributes the register-fragment loads across shared banks.
constexpr int kNativeStride = kNativeK / 4 + 1;

__device__ __forceinline__ void MmaInt8(int32_t* acc, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void CopyAsync16(void* destination, const void* source, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(source), "r"(valid ? 16 : 0) : "memory");
#endif
}

// Weights use the SM80 fpA_intB layout: 64 K x (16 / Bits) columns form one 128-byte tile. Block size is 32 for
// INT4/INT8 and 64 for INT2, so each scale interval covers one or two 32-wide MMA steps. Each warp owns eight columns
// and RowFragments 16-row fragments, so every decoded weight fragment feeds RowFragments MMAs. The scales of the
// next K tile are loaded ahead into registers to keep their latency off the promotion path.
template <typename T, typename ScaleT, int Bits, int FixedRows = 0, int RowFragments = 1>
__global__ void PrepackedMatMul(const int8_t* a, const float* a_scales, const uint8_t* b,
                                const void* b_scales, const T* bias, T* y,
                                bool per_tensor, int rows, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int TileM = 16 * RowFragments;
  constexpr int TileN = 64;
  constexpr int TileK = 128;
  constexpr int Stages = 2;
  constexpr int AStride = TileK / 4 + 4;
  constexpr int Interleave = 16 / Bits;
  constexpr int ColumnWords = 32 / Interleave;
  constexpr int ChunksPerColumnGroup = TileK / 64 * 8;
  constexpr int BlockSize = Bits == 2 ? 64 : 32;
  constexpr int StepsPerGroup = BlockSize / 32;
  constexpr int Steps = TileK / 32;
  constexpr int GroupsPerTile = TileK / BlockSize;
  __shared__ __align__(16) uint32_t shared_a[Stages][TileM][AStride];
  __shared__ __align__(16) uint4 shared_b[Stages][(TileN / Interleave) * ChunksPerColumnGroup];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int column_tiles = n / TileN;
  const int active_rows = FixedRows == 0 ? rows : FixedRows;
  const int tile_row = FixedRows == 0 ? (blockIdx.x / column_tiles) * TileM : 0;
  const int tile_column = (FixedRows == 0 ? blockIdx.x % column_tiles : blockIdx.x) * TileN;
  const int groups = k / BlockSize;
  float total[RowFragments][4];
  float activation_scale[RowFragments][2];
  bool normal_activation_scale[RowFragments][2];
#pragma unroll
  for (int rf = 0; rf < RowFragments; ++rf) {
    const int scale_row = tile_row + rf * 16 + lane % 16;
    const float row_scale = scale_row < active_rows ? a_scales[per_tensor ? 0 : scale_row] : 1.0f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      activation_scale[rf][half] = __shfl_sync(0xffffffff, row_scale, lane / 4 + half * 8);
      normal_activation_scale[rf][half] =
          isfinite(activation_scale[rf][half]) && fabsf(activation_scale[rf][half]) >= FLT_MIN;
    }
  }

  auto load = [&](int pos, int stage) {
#pragma unroll
    for (int i = threadIdx.x; i < TileM * (TileK / 16); i += 256) {
      const int row = i / (TileK / 16);
      const int offset = (i % (TileK / 16)) * 16;
      const bool valid = tile_row + row < active_rows && pos < k;
      CopyAsync16(&shared_a[stage][row][offset / 4],
                  valid ? a + int64_t{tile_row + row} * k + pos + offset : a, valid);
    }
#pragma unroll
    for (int i = threadIdx.x; i < (TileN / Interleave) * ChunksPerColumnGroup; i += 256) {
      const int column_group = i / ChunksPerColumnGroup;
      const int chunk = i % ChunksPerColumnGroup;
      const int64_t global_column_group = tile_column / Interleave + column_group;
      const int64_t source_chunk = global_column_group * (k / 8) + (pos / 64) * 8 + chunk;
      const bool valid = pos < k;
      CopyAsync16(&shared_b[stage][i],
                  valid ? &reinterpret_cast<const uint4*>(b)[source_chunk] : reinterpret_cast<const uint4*>(b), valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };

  // Lane l < 4 owns the scales of columns 2l and 2l+1 of this warp's eight; the others are shuffled from it.
  const int scale_column = tile_column + warp * 8 + (lane % 4) * 2;
  auto load_scales = [&](int pos, float (&s0)[GroupsPerTile], float (&s1)[GroupsPerTile]) {
#pragma unroll
    for (int g = 0; g < GroupsPerTile; ++g) {
      const int group = pos / BlockSize + g;
      const bool valid = lane < 4 && pos < k;
      s0[g] = valid ? LoadTypedScale<ScaleT>(b_scales, int64_t{scale_column} * groups + group) : 0.0f;
      s1[g] = valid ? LoadTypedScale<ScaleT>(b_scales, int64_t{scale_column + 1} * groups + group) : 0.0f;
    }
  };
  float current0[GroupsPerTile], current1[GroupsPerTile];
  float next0[GroupsPerTile], next1[GroupsPerTile];
  load_scales(0, current0, current1);

  load(0, 0);
  for (int pos = 0; pos < k; pos += TileK) {
    const int stage = (pos / TileK) % Stages;
    load(pos + TileK, (stage + 1) % Stages);
    load_scales(pos + TileK, next0, next1);
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();

    const int column = warp * 8 + lane / 4;
    const uint32_t* column_words = reinterpret_cast<const uint32_t*>(&shared_b[stage][0]) +
                                   (column / Interleave) * (ChunksPerColumnGroup * 4) +
                                   (column % Interleave) * ColumnWords;
    int32_t acc[RowFragments][4];
#pragma unroll
    for (int step = 0; step < Steps; ++step) {
      const uint32_t* tile_words = column_words + (step / 2) * 32;
      const uint32_t b0 = DecodePrepacked<Bits>(tile_words, step % 2, lane % 4, false);
      const uint32_t b1 = DecodePrepacked<Bits>(tile_words, step % 2, lane % 4, true);
      if (step % StepsPerGroup == 0) {
#pragma unroll
        for (int rf = 0; rf < RowFragments; ++rf) acc[rf][0] = acc[rf][1] = acc[rf][2] = acc[rf][3] = 0;
      }
#pragma unroll
      for (int rf = 0; rf < RowFragments; ++rf) {
        uint32_t fragment_a[4];
        const int row = rf * 16 + lane % 16;
        const uint32_t a_address = static_cast<uint32_t>(__cvta_generic_to_shared(
            &shared_a[stage][row][step * 8 + (lane / 16) * 4]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(fragment_a[0]), "=r"(fragment_a[1]),
                       "=r"(fragment_a[2]), "=r"(fragment_a[3]) : "r"(a_address));
        MmaInt8(acc[rf], fragment_a, b0, b1);
      }
      if (step % StepsPerGroup != StepsPerGroup - 1) continue;

      const int group_in_tile = step / StepsPerGroup;
      const float scale0 = __shfl_sync(0xffffffff, current0[group_in_tile], lane % 4);
      const float scale1 = __shfl_sync(0xffffffff, current1[group_in_tile], lane % 4);
#pragma unroll
      for (int rf = 0; rf < RowFragments; ++rf) {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
          const float term = DecodeScaleProduct<T>(
              acc[rf][element], activation_scale[rf][element / 2], element % 2 == 0 ? scale0 : scale1,
              normal_activation_scale[rf][element / 2]);
          total[rf][element] = pos == 0 && group_in_tile == 0 ? term : __fadd_rn(total[rf][element], term);
        }
      }
    }
#pragma unroll
    for (int g = 0; g < GroupsPerTile; ++g) {
      current0[g] = next0[g];
      current1[g] = next1[g];
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

#pragma unroll
  for (int rf = 0; rf < RowFragments; ++rf) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      const int row = tile_row + rf * 16 + lane / 4 + (element / 2) * 8;
      const int column = tile_column + warp * 8 + (lane % 4) * 2 + element % 2;
      if (row < active_rows) {
        y[int64_t{row} * n + column] =
            from_float<T>(__fadd_rn(total[rf][element], bias ? to_float(bias[column]) : 0.0f));
      }
    }
  }
#endif
}

// One common interval permits keeping only INT32 fragments until the epilogue.
// The aligned W8 tile also avoids per-code division, byte loads, and decoding.
template <typename T, int TileK>
__global__ void NativeSingleIntervalMatMul(const int8_t* a, const float* a_scales, const uint8_t* b,
                                           const void* b_scales, DynamicWeightScaleType scale_type,
                                           const T* bias, T* y, bool per_tensor, int64_t rows, int64_t n,
                                           int k, int64_t block_size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int TileM = 64;
  constexpr int Stages = 2;
  constexpr int tile_n = 128;
  constexpr int tile_k = TileK;
  constexpr int threads = 256;
  // Four-word padding keeps uint4 stores aligned and makes the eight rows x
  // four words in each MMA register load hit all 32 shared banks.
  constexpr int stride = tile_k / 4 + 4;
  constexpr int m_fragments = TileM / 32;
  extern __shared__ __align__(16) uint32_t storage[];
  auto shared_a = reinterpret_cast<uint32_t (*)[TileM][stride]>(storage);
  auto shared_b = reinterpret_cast<uint32_t (*)[tile_n][stride]>(storage + Stages * TileM * stride);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warp_row = (warp / 4) * (TileM / 2);
  const int warp_column = (warp % 4) * 32;
  const int64_t column_tiles = n / tile_n + (n % tile_n != 0);
  const int64_t row_tiles = rows / TileM + (rows % TileM != 0);
  for (int64_t tile = blockIdx.x; tile < row_tiles * column_tiles; tile += gridDim.x) {
    const int64_t tile_row = (tile / column_tiles) * TileM;
    const int64_t tile_column = (tile % column_tiles) * tile_n;
    int32_t acc[m_fragments][4][4] = {};
    auto load = [&](int pos, int stage) {
#pragma unroll
      for (int i = threadIdx.x; i < TileM * (tile_k / 16); i += threads) {
        const int local_row = i / (tile_k / 16);
        const int offset = (i % (tile_k / 16)) * 16;
        const int64_t row = tile_row + local_row;
        const bool valid = row < rows && pos + offset < k;
        CopyAsync16(&shared_a[stage][local_row][offset / 4],
                    valid ? a + row * k + pos + offset : a, valid);
      }
#pragma unroll
      for (int i = threadIdx.x; i < tile_n * (tile_k / 16); i += threads) {
        const int local_column = i / (tile_k / 16);
        const int offset = (i % (tile_k / 16)) * 16;
        const int64_t column = tile_column + local_column;
        const bool valid = column < n && pos + offset < k;
        CopyAsync16(&shared_b[stage][local_column][offset / 4],
                    valid ? b + column * block_size + pos + offset : b, valid);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    };
#pragma unroll
    for (int stage = 0; stage < Stages - 1; ++stage) load(stage * tile_k, stage);
    for (int pos = 0; pos < k; pos += tile_k) {
      const int stage = (pos / tile_k) % Stages;
      const int valid_k = min(tile_k, k - pos);
      load(pos + (Stages - 1) * tile_k, (stage + Stages - 1) % Stages);
      asm volatile("cp.async.wait_group %0;" ::"n"(Stages - 1) : "memory");
      __syncthreads();
#pragma unroll
      for (int half_k = 0; half_k < tile_k / 32; ++half_k) {
        if (half_k * 32 < valid_k) {
          uint32_t fragment_b[4][2];
#pragma unroll
          for (int cf = 0; cf < 4; ++cf) {
            const int column = warp_column + cf * 8 + lane % 8;
            const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
                &shared_b[stage][column][half_k * 8 + ((lane / 8) % 2) * 4]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                         : "=r"(fragment_b[cf][0]), "=r"(fragment_b[cf][1]) : "r"(address));
            fragment_b[cf][0] ^= 0x80808080u;
            fragment_b[cf][1] ^= 0x80808080u;
          }
#pragma unroll
          for (int mf = 0; mf < m_fragments; ++mf) {
            const int row = warp_row + mf * 16 + lane % 16;
            const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
                &shared_a[stage][row][half_k * 8 + (lane / 16) * 4]));
            uint32_t fragment_a[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(fragment_a[0]), "=r"(fragment_a[1]),
                           "=r"(fragment_a[2]), "=r"(fragment_a[3]) : "r"(address));
#pragma unroll
            for (int cf = 0; cf < 4; ++cf) {
              MmaInt8(acc[mf][cf], fragment_a, fragment_b[cf][0], fragment_b[cf][1]);
            }
          }
        }
      }
      __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float weight_scales[4][2];
#pragma unroll
    for (int cf = 0; cf < 4; ++cf) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int64_t column = tile_column + warp_column + cf * 8 + (lane % 4) * 2 + c;
        weight_scales[cf][c] = column < n ? LoadScale(b_scales, scale_type, column) : 0.0f;
      }
    }
#pragma unroll
    for (int mf = 0; mf < m_fragments; ++mf) {
      float activation_scales[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int64_t row = tile_row + warp_row + mf * 16 + lane / 4 + r * 8;
        activation_scales[r] = row < rows ? a_scales[per_tensor ? 0 : row] : 1.0f;
      }
#pragma unroll
      for (int cf = 0; cf < 4; ++cf) {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
          const int64_t row = tile_row + warp_row + mf * 16 + lane / 4 + (element / 2) * 8;
          const int64_t column = tile_column + warp_column + cf * 8 + (lane % 4) * 2 + element % 2;
          if (row < rows && column < n) {
            const float scaled = ScaleProduct(acc[mf][cf][element], activation_scales[element / 2],
                                              weight_scales[cf][element % 2]);
            y[row * n + column] = from_float<T>(__fadd_rn(scaled, bias ? to_float(bias[column]) : 0.0f));
          }
        }
      }
    }
  }
#endif
}

template <typename T, int Bits, int TileM, bool Common128, int TileN, int Stages, int WarpsM, int TileK, bool RowGuard,
          bool Fine>
__device__ __forceinline__ void NativeAlignedGroupedBody(const int8_t* a, const float* a_scales, const uint8_t* b,
                                                         const void* b_scales, DynamicWeightScaleType scale_type,
                                                         const T* bias, T* y, bool per_tensor, int rows, int n,
                                                         int k, int runtime_block_size,
                                                         int runtime_activation_block_size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // Fine promotes after every 32-wide MMA step, for 32-aligned (not 64-aligned) scale intervals.
  // RowGuard keeps the fixed-geometry Common128 body but predicates rows for a partial last row tile.
  constexpr bool full_rows = Common128 && !RowGuard;
  const int block_size = Common128 ? 128 : runtime_block_size;
  const int activation_block_size = Common128 ? k : runtime_activation_block_size;
  constexpr int tile_n = TileN;
  constexpr int pack = 8 / Bits;
  constexpr int tile_k = TileK;
  constexpr int threads = 128;
  // Four-word padding keeps uint4 stores aligned and makes the eight rows x
  // four words in each MMA register load hit all 32 shared banks.
  constexpr int stride = tile_k / 4 + 4;
  constexpr int WarpsN = 4 / WarpsM;
  constexpr int m_fragments = TileM / (16 * WarpsM);
  constexpr int b_stride = (tile_k / (4 * pack)) | 4;
  constexpr int n_fragments = TileN / (8 * WarpsN);
  const int groups = k / block_size + (k % block_size != 0);
  const int activation_groups = per_tensor ? 0 : k / activation_block_size + (k % activation_block_size != 0);
  __shared__ __align__(16) uint32_t shared_a[Stages][TileM][stride];
  __shared__ __align__(16) uint32_t shared_b[Stages][tile_n][b_stride];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warp_row = (warp / WarpsN) * (TileM / WarpsM);
  const int warp_column = (warp % WarpsN) * (TileN / WarpsN);
  const int column_tiles = n / tile_n + (n % tile_n != 0);
  const int row_tiles = rows / TileM + (rows % TileM != 0);
  for (int64_t tile = blockIdx.x; tile < int64_t{row_tiles} * column_tiles; tile += gridDim.x) {
    const int tile_row = (tile / column_tiles) * TileM;
    const int tile_column = (tile % column_tiles) * tile_n;
    int32_t acc[m_fragments][n_fragments][4] = {};
    float total[m_fragments][n_fragments][4];
    float row_scales[m_fragments];
    int cached_activation_group = -1;
    int weight_group = 0;
    int activation_group = 0;
    int weight_end = min(k, block_size);
    int activation_end = min(k, activation_block_size);
    int interval_start = 0;
    int interval_end = min(weight_end, activation_end);
    if constexpr (Common128) {
#pragma unroll
      for (int mf = 0; mf < m_fragments; ++mf) {
        const int scale_row = tile_row + warp_row + mf * 16 + lane % 16;
        row_scales[mf] = (full_rows || scale_row < rows) ? a_scales[per_tensor ? 0 : scale_row] : 1.0f;
      }
    }
    // The weight scale of the next interval is fetched right after the previous promotion so its
    // global-memory latency overlaps the MMAs instead of stalling the promotion.
    const int scale_column = tile_column + warp_column + lane;
    auto load_weight_scale = [&](int group) {
      return (lane < TileN / WarpsN && (Common128 || scale_column < n) && group < groups)
                 ? LoadScale(b_scales, scale_type, int64_t{scale_column} * groups + group)
                 : 0.0f;
    };
    float weight_scale = load_weight_scale(0);
    auto load = [&](int pos, int stage) {
#pragma unroll
      for (int i = threadIdx.x; i < TileM * (tile_k / 16); i += threads) {
        const int local_row = i / (tile_k / 16);
        const int offset = (i % (tile_k / 16)) * 16;
        const int row = tile_row + local_row;
        const bool valid = (full_rows || row < rows) && pos + offset < k;
        CopyAsync16(&shared_a[stage][local_row][offset / 4],
                    valid ? a + int64_t{row} * k + pos + offset : a, valid);
      }
#pragma unroll
      for (int i = threadIdx.x; i < tile_n * (tile_k / (16 * pack)); i += threads) {
        const int local_column = i / (tile_k / (16 * pack));
        const int offset = (i % (tile_k / (16 * pack))) * 16;
        const int column = tile_column + local_column;
        const bool valid = (Common128 || column < n) && pos + offset * pack < k;
        CopyAsync16(&shared_b[stage][local_column][offset / 4],
                    valid ? b + int64_t{column} * groups * (block_size / pack) + pos / pack + offset : b, valid);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    };
#pragma unroll
    for (int stage = 0; stage < Stages - 1; ++stage) load(stage * tile_k, stage);
    for (int pos = 0; pos < k; pos += tile_k) {
      const int stage = (pos / tile_k) % Stages;
      const int valid_k = min(tile_k, k - pos);
      load(pos + (Stages - 1) * tile_k, (stage + Stages - 1) % Stages);
      asm volatile("cp.async.wait_group %0;" ::"n"(Stages - 1) : "memory");
      __syncthreads();
      uint32_t fragment_a[2][m_fragments][4];
      uint32_t fragment_b[2][n_fragments][2];
      auto load_fragments = [&](int half_k, int buffer) {
#pragma unroll
        for (int cf = 0; cf < n_fragments; ++cf) {
          const int column = warp_column + cf * 8 + lane / 4;
          const auto* packed = reinterpret_cast<const uint8_t*>(&shared_b[stage][column][0]);
          fragment_b[buffer][cf][0] = DecodeWeight4<Bits>(packed, half_k * 32 + (lane % 4) * 4);
          fragment_b[buffer][cf][1] = DecodeWeight4<Bits>(packed, half_k * 32 + (lane % 4) * 4 + 16);
        }
#pragma unroll
        for (int mf = 0; mf < m_fragments; ++mf) {
          const int row = warp_row + mf * 16 + lane % 16;
          const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
              &shared_a[stage][row][half_k * 8 + (lane / 16) * 4]));
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                       : "=r"(fragment_a[buffer][mf][0]), "=r"(fragment_a[buffer][mf][1]),
                         "=r"(fragment_a[buffer][mf][2]), "=r"(fragment_a[buffer][mf][3]) : "r"(address));
        }
      };
      float fine_scales[Fine ? tile_k / 32 : 1];
      if constexpr (Fine) {
#pragma unroll
        for (int h = 0; h < tile_k / 32; ++h) fine_scales[h] = load_weight_scale((pos + h * 32) / block_size);
      }
      auto promote = [&](int end, float weight_scale_in) {
        float weight_scales[n_fragments][2];
#pragma unroll
        for (int cf = 0; cf < n_fragments; ++cf) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            weight_scales[cf][c] = __shfl_sync(0xffffffff, weight_scale_in, cf * 8 + (lane % 4) * 2 + c);
          }
        }
        if (!Common128 && activation_group != cached_activation_group) {
#pragma unroll
          for (int mf = 0; mf < m_fragments; ++mf) {
            const int scale_row = tile_row + warp_row + mf * 16 + lane % 16;
            row_scales[mf] =
                scale_row < rows
                    ? a_scales[per_tensor ? 0 : int64_t{scale_row} * activation_groups + activation_group]
                    : 1.0f;
          }
          cached_activation_group = activation_group;
        }
#pragma unroll
        for (int mf = 0; mf < m_fragments; ++mf) {
          float activation_scales[2];
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            activation_scales[r] = __shfl_sync(0xffffffff, row_scales[mf], lane / 4 + r * 8);
          }
#pragma unroll
          for (int cf = 0; cf < n_fragments; ++cf) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
              const int row = tile_row + warp_row + mf * 16 + lane / 4 + (element / 2) * 8;
              const int column = tile_column + warp_column + cf * 8 + (lane % 4) * 2 + element % 2;
              if (full_rows || (row < rows && (Common128 || column < n))) {
                const float term = ScaleProduct<std::is_same_v<T, half>>(acc[mf][cf][element],
                                                                         activation_scales[element / 2],
                                                                         weight_scales[cf][element % 2]);
                total[mf][cf][element] = (Common128 ? end == 128 : interval_start == 0)
                                             ? term
                                             : __fadd_rn(total[mf][cf][element], term);
              }
              acc[mf][cf][element] = 0;
            }
          }
        }
        if (Common128) {
          weight_scale = load_weight_scale(end / 128);
        } else if (interval_end == weight_end) {
          ++weight_group;
          weight_end = interval_end + min(k - interval_end, block_size);
          if constexpr (!Fine) weight_scale = load_weight_scale(weight_group);
        }
        if (!Common128 && interval_end == activation_end) {
          ++activation_group;
          activation_end = interval_end + min(k - interval_end, activation_block_size);
        }
        interval_start = interval_end;
        interval_end = min(weight_end, activation_end);
      };
      load_fragments(0, 0);
#pragma unroll
      for (int half_k = 0; half_k < tile_k / 32; ++half_k) {
        if (half_k * 32 < valid_k) {
          if (half_k + 1 < tile_k / 32) load_fragments(half_k + 1, (half_k + 1) % 2);
#pragma unroll
          for (int mf = 0; mf < m_fragments; ++mf) {
#pragma unroll
            for (int cf = 0; cf < n_fragments; ++cf) {
              MmaInt8(acc[mf][cf], fragment_a[half_k % 2][mf],
                      fragment_b[half_k % 2][cf][0], fragment_b[half_k % 2][cf][1]);
            }
          }
          if constexpr (Fine) {
            if (pos + half_k * 32 + 32 == interval_end) promote(interval_end, fine_scales[half_k]);
          }
        }
      }
      if constexpr (!Fine) {
        if (Common128 ? (pos + tile_k) % 128 == 0 : pos + tile_k == interval_end) promote(pos + tile_k, weight_scale);
      }
      __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int mf = 0; mf < m_fragments; ++mf) {
#pragma unroll
      for (int cf = 0; cf < n_fragments; ++cf) {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
          const int row = tile_row + warp_row + mf * 16 + lane / 4 + (element / 2) * 8;
          const int column = tile_column + warp_column + cf * 8 + (lane % 4) * 2 + element % 2;
          if (full_rows || (row < rows && (Common128 || column < n))) {
            y[int64_t{row} * n + column] = from_float<T>(__fadd_rn(total[mf][cf][element],
                                                                   bias ? to_float(bias[column]) : 0.0f));
          }
        }
      }
    }
  }
#endif
}

template <typename T, int Bits, int TileM, bool Common128 = false, int TileN = 64, int Stages = 3, int WarpsM = 2,
          int TileK = 64, bool RowGuard = false, bool Fine = false>
__global__ void NativeAlignedGroupedMatMul(const int8_t* a, const float* a_scales, const uint8_t* b,
                                           const void* b_scales, DynamicWeightScaleType scale_type,
                                           const T* bias, T* y, bool per_tensor, int rows, int n,
                                           int k, int runtime_block_size, int runtime_activation_block_size) {
  NativeAlignedGroupedBody<T, Bits, TileM, Common128, TileN, Stages, WarpsM, TileK, RowGuard, Fine>(
      a, a_scales, b, b_scales, scale_type, bias, y, per_tensor, rows, n, k, runtime_block_size,
      runtime_activation_block_size);
}

// The runtime-interval body needs the register cap to keep three 128-thread CTAs per SM; the fixed Common128 body
// is slower with any register attribute, so it keeps the compiler default.
template <typename T, int Bits, int TileM, bool Common128 = false, int TileN = 64, int Stages = 3, int WarpsM = 2,
          int TileK = 64, bool RowGuard = false, bool Fine = false>
__global__ void __launch_bounds__(128, 3)
    NativeAlignedGroupedMatMulCapped(const int8_t* a, const float* a_scales, const uint8_t* b,
                                     const void* b_scales, DynamicWeightScaleType scale_type,
                                     const T* bias, T* y, bool per_tensor, int rows, int n,
                                     int k, int runtime_block_size, int runtime_activation_block_size) {
  NativeAlignedGroupedBody<T, Bits, TileM, Common128, TileN, Stages, WarpsM, TileK, RowGuard, Fine>(
      a, a_scales, b, b_scales, scale_type, bias, y, per_tensor, rows, n, k, runtime_block_size,
      runtime_activation_block_size);
}

template <typename T, int Bits>
__global__ void NativeIntegerMatMul(const int8_t* a, const float* a_scales, const uint8_t* b,
                                    const void* b_scales, DynamicWeightScaleType scale_type,
                                    const T* bias, T* y, bool per_tensor, int64_t rows, int64_t n, int64_t k,
                                    int64_t block_size, int64_t activation_block_size) {
// The host checks both the runtime architecture and the loaded image's PTX
// target. In particular, JIT of a pre-SM80 image must not select this body.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __shared__ uint32_t shared_a[kNativeM][kNativeStride];
  __shared__ uint32_t shared_b[kNativeN][kNativeStride];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int fragment_row = (warp / 2) * 16 + lane / 4;
  const int fragment_column = (warp % 2) * 32 + (lane % 4) * 2;
  const int64_t groups = k / block_size + (k % block_size != 0);
  const int64_t activation_groups = per_tensor ? 0 : k / activation_block_size + (k % activation_block_size != 0);
  const int64_t blob_size = block_size / (8 / Bits);
  const int64_t column_tiles = n / kNativeN + (n % kNativeN != 0);
  const int64_t row_tiles = rows / kNativeM + (rows % kNativeM != 0);
  for (int64_t tile = blockIdx.x; tile < row_tiles * column_tiles; tile += gridDim.x) {
    const int64_t tile_row = (tile / column_tiles) * kNativeM;
    const int64_t tile_column = (tile % column_tiles) * kNativeN;
    float total[4][4];
    for (int64_t start = 0; start < k;) {
      const int64_t weight_group = start / block_size;
      const int64_t activation_group = start / activation_block_size;
      const int64_t length = min(k - start, min(block_size - start % block_size,
                                                activation_block_size - start % activation_block_size));
      int32_t acc[4][4] = {};
      for (int64_t offset = 0; offset < length; offset += kNativeK) {
        const int valid_k = static_cast<int>(min(int64_t{kNativeK}, length - offset));
        const int64_t pos = start + offset;
        // Cooperative four-byte loads, with no expanded global weight matrix.
        // Predication excludes serialized padding, including unused packed bits.
        for (int i = threadIdx.x; i < kNativeM * (kNativeK / 4); i += kNativeThreads) {
          const int local_row = i / (kNativeK / 4);
          const int word = i % (kNativeK / 4);
          const int64_t row = tile_row + local_row;
          const int local_k = word * 4;
          uint32_t codes = 0;
          if (row < rows && local_k < valid_k) {
            const int8_t* input = a + row * k + pos + local_k;
            if (local_k + 4 <= valid_k && (row * k + pos + local_k) % 4 == 0) {
              codes = *reinterpret_cast<const uint32_t*>(input);
            } else {
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                if (local_k + j < valid_k) {
                  codes |= static_cast<uint32_t>(static_cast<uint8_t>(input[j])) << (8 * j);
                }
              }
            }
          }
          shared_a[local_row][word] = codes;
        }
        for (int i = threadIdx.x; i < kNativeN * (kNativeK / 4); i += kNativeThreads) {
          const int local_column = i / (kNativeK / 4);
          const int word = i % (kNativeK / 4);
          const int64_t column = tile_column + local_column;
          const int local_k = word * 4;
          uint32_t codes = 0;
          if (column < n && local_k < valid_k) {
            const uint8_t* blob = b + (column * groups + weight_group) * blob_size;
            const int64_t j = pos % block_size + local_k;
            if (local_k + 4 <= valid_k) {
              codes = DecodeWeight4<Bits>(blob, j);
            } else {
#pragma unroll
              for (int t = 0; t < 4; ++t) {
                if (local_k + t < valid_k) {
                  codes |= static_cast<uint32_t>(static_cast<uint8_t>(DecodeWeight<Bits>(blob, j + t))) << (8 * t);
                }
              }
            }
          }
          shared_b[local_column][word] = codes;
        }
        __syncthreads();
#pragma unroll
        for (int half_k = 0; half_k < 2; ++half_k) {
          if (half_k * 32 < valid_k) {
            const int word = half_k * 8 + lane % 4;
            const uint32_t fragment_a[4] = {
                shared_a[fragment_row][word],
                shared_a[fragment_row + 8][word],
                shared_a[fragment_row][word + 4],
                shared_a[fragment_row + 8][word + 4]};
#pragma unroll
            for (int column_fragment = 0; column_fragment < 4; ++column_fragment) {
              const int column = (warp % 2) * 32 + column_fragment * 8 + lane / 4;
              MmaInt8(acc[column_fragment], fragment_a, shared_b[column][word], shared_b[column][word + 4]);
            }
          }
        }
        __syncthreads();
      }
      // Reset INT32 at every common boundary, then add terms strictly in K
      // order. Equal adjacent scales do not permit merging interval dots.
#pragma unroll
      for (int column_fragment = 0; column_fragment < 4; ++column_fragment) {
#pragma unroll
        for (int element = 0; element < 4; ++element) {
          const int64_t row = tile_row + fragment_row + (element / 2) * 8;
          const int64_t column = tile_column + fragment_column + column_fragment * 8 + element % 2;
          if (row < rows && column < n) {
            const float term = ScaleProduct<std::is_same_v<T, half>>(acc[column_fragment][element],
                                                                     a_scales[per_tensor ? 0 : row * activation_groups + activation_group],
                                                                     LoadScale(b_scales, scale_type, column * groups + weight_group));
            total[column_fragment][element] = start == 0 ? term : __fadd_rn(total[column_fragment][element], term);
          }
        }
      }
      start += length;
    }
#pragma unroll
    for (int column_fragment = 0; column_fragment < 4; ++column_fragment) {
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        const int64_t row = tile_row + fragment_row + (element / 2) * 8;
        const int64_t column = tile_column + fragment_column + column_fragment * 8 + element % 2;
        if (row < rows && column < n) {
          y[row * n + column] = from_float<T>(__fadd_rn(total[column_fragment][element],
                                                        bias ? to_float(bias[column]) : 0.0f));
        }
      }
    }
  }
#endif
}

template <typename Kernel>
Status NativeKernelAvailable(Kernel kernel, bool& available) {
  available = false;
  cudaFuncAttributes attributes{};
  const auto error = cudaFuncGetAttributes(&attributes, kernel);
  if (error == cudaErrorNoKernelImageForDevice || error == cudaErrorInvalidDeviceFunction) {
    // Availability failure is not a compute failure. Clear it before generic launch.
    (void)cudaGetLastError();
    return Status::OK();
  }
  CUDA_RETURN_IF_ERROR(error);
  // A pre-SM80 PTX image can JIT to SM80+, but its source was compiled without
  // the guarded MMA body. Check the source target as well as the loaded binary.
  available = attributes.binaryVersion >= 80 && attributes.ptxVersion >= 80;
  return Status::OK();
}

// Row fragments per CTA of the prepacked kernel. On A100 a decoded weight fragment serving 2-4 row fragments
// raised M >= 128 throughput by up to 1.9x. Use the largest count that does not exceed the row count and still
// leaves a CTA per SM (96 of 108), and at most two for INT4, whose nibble decode is the costliest and which gained nothing from four. Other
// architectures keep one fragment until measured.
int PrepackedRowFragments(int bits, int64_t rows, int64_t n, int compute_capability) {
  if (compute_capability != 80 || rows <= 16) return 1;
  constexpr int64_t kMinimumCtas = 96;
  for (int fragments = bits == 4 ? 2 : 4; fragments > 1; fragments /= 2) {
    const int64_t tile_rows = 16 * fragments;
    if (tile_rows <= rows && (rows + tile_rows - 1) / tile_rows * (n / 64) >= kMinimumCtas) return fragments;
  }
  return 1;
}

template <typename T, typename ScaleT, int Bits>
Status LaunchPrepacked(const int8_t* a, const float* a_scales, const uint8_t* b,
                       const void* b_scales, const void* bias, void* y,
                       bool per_tensor, int64_t rows, int64_t n, int64_t k,
                       int compute_capability, cudaStream_t stream) {
  const int fragments = PrepackedRowFragments(Bits, rows, n, compute_capability);
  auto kernel = rows == 8        ? PrepackedMatMul<T, ScaleT, Bits, 8>
                : rows == 9      ? PrepackedMatMul<T, ScaleT, Bits, 9>
                : fragments == 4 ? PrepackedMatMul<T, ScaleT, Bits, 0, 4>
                : fragments == 2 ? PrepackedMatMul<T, ScaleT, Bits, 0, 2>
                                 : PrepackedMatMul<T, ScaleT, Bits, 0, 1>;
  bool available;
  ORT_RETURN_IF_ERROR(NativeKernelAvailable(kernel, available));
  ORT_RETURN_IF_NOT(available, "The SM80+ integer MMA body is unavailable in the loaded CUDA image.");
  const int64_t column_tiles = n / 64;
  const int64_t tile_rows = 16 * fragments;
  const int64_t row_tiles = rows == 8 || rows == 9 ? 1 : rows / tile_rows + (rows % tile_rows != 0);
  const int64_t tiles = SafeInt<int64_t>(column_tiles) * row_tiles;
  ORT_RETURN_IF_NOT(tiles <= std::numeric_limits<int>::max(), "Prepacked tile grid exceeds CUDA limits.");
  kernel<<<static_cast<int>(tiles), 256, 0, stream>>>(
      a, a_scales, b, b_scales, static_cast<const T*>(bias), static_cast<T*>(y),
      per_tensor, static_cast<int>(rows), static_cast<int>(n), static_cast<int>(k));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

// CTA tile shapes of the aligned grouped kernel. The 16-row tiles give all four warps distinct columns, and
// k128 variants stage 128-byte activation rows, which suits Ampere's L2 sectors better than 64-byte rows.
enum class GroupedTile { k16x32,
                         k16x64,
                         k32x64,
                         k64x64 };

constexpr int GroupedTileRows(GroupedTile tile) {
  return tile == GroupedTile::k16x32 || tile == GroupedTile::k16x64 ? 16 : tile == GroupedTile::k32x64 ? 32
                                                                                                       : 64;
}

constexpr int GroupedTileColumns(GroupedTile tile) { return tile == GroupedTile::k16x32 ? 32 : 64; }

template <typename T, int Bits, int TileM, bool Common128, int TileN, int Stages, int WarpsM, int TileK,
          bool RowGuard, bool Fine>
constexpr auto GroupedKernel() {
  if constexpr (Common128) {
    return NativeAlignedGroupedMatMul<T, Bits, TileM, Common128, TileN, Stages, WarpsM, TileK, RowGuard, Fine>;
  } else {
    return NativeAlignedGroupedMatMulCapped<T, Bits, TileM, Common128, TileN, Stages, WarpsM, TileK, RowGuard, Fine>;
  }
}

template <typename T, int Bits, bool Common128, bool RowGuard, bool Fine = false>
static auto PickGroupedKernel(GroupedTile tile, bool k128) {
  // Fine intervals never use 128-wide K tiles; equal arguments keep a single instantiation.
  constexpr int K128 = Fine ? 64 : 128;
  switch (tile) {
    case GroupedTile::k16x32:
      return k128 ? GroupedKernel<T, Bits, 16, Common128, 32, 3, 1, K128, RowGuard, Fine>()
                  : GroupedKernel<T, Bits, 16, Common128, 32, 3, 1, 64, RowGuard, Fine>();
    case GroupedTile::k16x64:
      return k128 ? GroupedKernel<T, Bits, 16, Common128, 64, 3, 1, K128, RowGuard, Fine>()
                  : GroupedKernel<T, Bits, 16, Common128, 64, 3, 1, 64, RowGuard, Fine>();
    case GroupedTile::k32x64:
      return k128 ? GroupedKernel<T, Bits, 32, Common128, 64, 3, 2, K128, RowGuard, Fine>()
                  : GroupedKernel<T, Bits, 32, Common128, 64, 3, 2, 64, RowGuard, Fine>();
    case GroupedTile::k64x64:
      // Two 128-wide stages already fill the 48 KB static shared-memory limit for W8.
      return k128 ? GroupedKernel<T, Bits, 64, Common128, 64, 2, 2, K128, RowGuard, Fine>()
                  : GroupedKernel<T, Bits, 64, Common128, 64, 3, 2, 64, RowGuard, Fine>();
  }
  return GroupedKernel<T, Bits, 32, Common128, 64, 3, 2, 64, RowGuard, Fine>();
}

// Ampere (SM80) selection measured on A100 (108 SMs): use the largest tile that wastes few padded rows and still
// leaves about 1.5 CTAs per SM, otherwise the narrow 16x32 tile. Other architectures keep the H200 rule.
GroupedTile SelectGroupedTile(int64_t rows, int64_t column_tiles_64, int compute_capability) {
  if (compute_capability == 80) {
    constexpr int64_t kMinimumCtas = 160;
    for (GroupedTile tile : {GroupedTile::k64x64, GroupedTile::k32x64, GroupedTile::k16x64}) {
      const int64_t tile_rows = GroupedTileRows(tile);
      const int64_t row_tiles = (rows + tile_rows - 1) / tile_rows;
      if (row_tiles * tile_rows * 10 <= rows * 13 && row_tiles * column_tiles_64 >= kMinimumCtas) return tile;
    }
    return GroupedTile::k16x32;
  }
  const size_t large_tiles = SafeInt<size_t>(rows / 64 + (rows % 64 != 0)) * column_tiles_64;
  const size_t large_tile_threshold = compute_capability >= 120 ? 128 : 256;
  return large_tiles >= large_tile_threshold ? GroupedTile::k64x64 : GroupedTile::k32x64;
}

template <typename T, int Bits>
Status LaunchMatMul(const int8_t* a, const float* a_scales, const uint8_t* b,
                    const void* b_scales, DynamicWeightScaleType scale_type, const void* bias, void* y,
                    bool per_tensor, int64_t rows, int64_t n, int64_t k, int64_t block_size,
                    int64_t activation_block_size, int64_t weight_prepacked,
                    int compute_capability, cudaStream_t stream) {
  if (weight_prepacked != 0) {
    const int64_t prepacked_block = Bits == 2 ? 64 : 32;
    const bool supported = weight_prepacked == 1 && compute_capability >= 80 && rows >= 1 &&
                           rows <= std::numeric_limits<int>::max() && n >= 64 &&
                           n <= std::numeric_limits<int>::max() && k >= 128 &&
                           k <= std::numeric_limits<int>::max() && n % 64 == 0 && k % 128 == 0 &&
                           block_size == prepacked_block && activation_block_size >= k;
    ORT_RETURN_IF_NOT(
        supported,
        "weight_prepacked=1 requires SM80+ INT2 block-64, INT4 block-32, or INT8 block-32 execution, "
        "positive M, K divisible by 128, N divisible by 64, and one activation-scale interval.");
    switch (scale_type) {
      case DynamicWeightScaleType::Float:
        return LaunchPrepacked<T, float, Bits>(
            a, a_scales, b, b_scales, bias, y, per_tensor, rows, n, k, compute_capability, stream);
      case DynamicWeightScaleType::Half:
        return LaunchPrepacked<T, half, Bits>(
            a, a_scales, b, b_scales, bias, y, per_tensor, rows, n, k, compute_capability, stream);
      case DynamicWeightScaleType::BFloat16:
        return LaunchPrepacked<T, __nv_bfloat16, Bits>(
            a, a_scales, b, b_scales, bias, y, per_tensor, rows, n, k, compute_capability, stream);
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unsupported prepacked DynamicQuantizeMatMulNBits geometry.");
  }
  // Whole 64-code intervals permit aligned asynchronous copies even for W2.
  // Keep address products wide, while bounding the fast path's loop counters.
  // Consumer Blackwell's measured crossover is eight rows and A100's is one row; other architectures retain 32.
  const int64_t minimum_rows = compute_capability >= 120 ? 8 : compute_capability == 80 ? 1
                                                                                        : 32;
  const bool grouped_geometry =
      compute_capability >= 80 && rows >= minimum_rows && n >= 8 && k >= 64 && k % 64 == 0 &&
      rows <= std::numeric_limits<int>::max() - 128 && n <= std::numeric_limits<int>::max() - 128 &&
      k <= std::numeric_limits<int>::max() - 128 && block_size <= std::numeric_limits<int>::max() &&
      activation_block_size <= std::numeric_limits<int>::max() && block_size % 64 == 0 &&
      (activation_block_size >= k || activation_block_size % 64 == 0);
  // 32-aligned intervals (block 32, activation blocks such as 96) promote after every MMA step. Only measured on A100.
  const bool fine_geometry =
      compute_capability == 80 && rows >= 1 && n >= 8 && k >= 64 && k % 64 == 0 &&
      rows <= std::numeric_limits<int>::max() - 128 && n <= std::numeric_limits<int>::max() - 128 &&
      k <= std::numeric_limits<int>::max() - 128 && block_size <= std::numeric_limits<int>::max() &&
      activation_block_size <= std::numeric_limits<int>::max() && block_size % 32 == 0 &&
      (activation_block_size >= k || activation_block_size % 32 == 0) && !grouped_geometry;
  // On A100 the grouped tiles also beat the single-interval tile until the rows amortize its 64x128 shape.
  const bool grouped_before_single = compute_capability == 80 && rows < 640 && grouped_geometry;
  if constexpr (Bits == 8) {
    // K divisible by 16 keeps every qA row and packed-weight vector aligned.
    if (compute_capability >= 80 && rows >= 16 && !grouped_before_single && n >= 8 && k >= 32 && k % 16 == 0 &&
        k <= std::numeric_limits<int>::max() - 256 && block_size >= k && activation_block_size >= k) {
      const bool large_k_tile = compute_capability >= 120 ||
                                (compute_capability >= 90 && rows >= 128 && k >= 128) ||
                                (compute_capability == 80 && rows >= 640 && k >= 128);
      bool available;
      ORT_RETURN_IF_ERROR(NativeKernelAvailable(large_k_tile ? NativeSingleIntervalMatMul<T, 128>
                                                             : NativeSingleIntervalMatMul<T, 64>,
                                                available));
      if (available) {
        const int64_t row_tiles = rows / 64 + (rows % 64 != 0);
        const int64_t column_tiles = n / 128 + (n % 128 != 0);
        const size_t tiles = SafeInt<size_t>(row_tiles) * column_tiles;
        const int blocks = static_cast<int>(std::min<size_t>(65535, tiles));
        if (large_k_tile) {
          constexpr int shared_bytes = 2 * (64 + 128) * (128 / 4 + 4) * sizeof(uint32_t);
          CUDA_RETURN_IF_ERROR(cudaFuncSetAttribute(NativeSingleIntervalMatMul<T, 128>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
          NativeSingleIntervalMatMul<T, 128><<<blocks, 256, shared_bytes, stream>>>(
              a, a_scales, b, b_scales, scale_type, static_cast<const T*>(bias), static_cast<T*>(y),
              per_tensor, rows, n, static_cast<int>(k), block_size);
        } else {
          constexpr int shared_bytes = 2 * (64 + 128) * (64 / 4 + 4) * sizeof(uint32_t);
          NativeSingleIntervalMatMul<T, 64><<<blocks, 256, shared_bytes, stream>>>(
              a, a_scales, b, b_scales, scale_type, static_cast<const T*>(bias), static_cast<T*>(y),
              per_tensor, rows, n, static_cast<int>(k), block_size);
        }
        CUDA_RETURN_IF_ERROR(cudaGetLastError());
        return Status::OK();
      }
    }
  }
  if (grouped_geometry || fine_geometry) {
    const int64_t column_tiles_64 = n / 64 + (n % 64 != 0);
    const GroupedTile tile = SelectGroupedTile(rows, column_tiles_64, compute_capability);
    const int tile_m = GroupedTileRows(tile);
    const int tile_n = GroupedTileColumns(tile);
    const int64_t column_tiles = n / tile_n + (n % tile_n != 0);
    // A 128-code K tile needs every interval boundary and the K tail on a 128-code boundary.
    const bool k128 = compute_capability == 80 && k % 128 == 0 && block_size % 128 == 0 &&
                      (activation_block_size >= k || activation_block_size % 128 == 0);
    const bool common128 = block_size == 128 && k % 128 == 0 && activation_block_size >= k && n % tile_n == 0;
    // Predicating a partial last row tile of the fixed-geometry kernel beats the generic kernel on A100.
    const bool common128_rows = rows % tile_m == 0 || compute_capability == 80;
    auto kernel = fine_geometry                   ? PickGroupedKernel<T, Bits, false, false, true>(tile, false)
                  : !common128 || !common128_rows ? PickGroupedKernel<T, Bits, false, false>(tile, k128)
                  : rows % tile_m != 0            ? PickGroupedKernel<T, Bits, true, true>(tile, k128)
                                                  : PickGroupedKernel<T, Bits, true, false>(tile, k128);
    bool available;
    ORT_RETURN_IF_ERROR(NativeKernelAvailable(kernel, available));
    if (available) {
      const size_t tiles = SafeInt<size_t>(rows / tile_m + (rows % tile_m != 0)) * column_tiles;
      const int blocks = static_cast<int>(std::min<size_t>(65535, tiles));
      kernel<<<blocks, 128, 0, stream>>>(
          a, a_scales, b, b_scales, scale_type, static_cast<const T*>(bias), static_cast<T*>(y),
          per_tensor, static_cast<int>(rows), static_cast<int>(n), static_cast<int>(k),
          static_cast<int>(block_size), static_cast<int>(activation_block_size));
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      return Status::OK();
    }
  }
  const bool aligned_intervals = (block_size >= k || block_size % 32 == 0) &&
                                 (activation_block_size >= k || activation_block_size % 32 == 0);
  if (compute_capability >= 80 && rows >= 32 && n >= 8 && k >= 32 && aligned_intervals) {
    bool available;
    ORT_RETURN_IF_ERROR(NativeKernelAvailable(NativeIntegerMatMul<T, Bits>, available));
    if (available) {
      const int64_t row_tiles = rows / kNativeM + (rows % kNativeM != 0);
      const int64_t column_tiles = n / kNativeN + (n % kNativeN != 0);
      const size_t tiles = SafeInt<size_t>(row_tiles) * column_tiles;
      const int blocks = static_cast<int>(std::min<size_t>(65535, tiles));
      NativeIntegerMatMul<T, Bits><<<blocks, kNativeThreads, 0, stream>>>(
          a, a_scales, b, b_scales, scale_type, static_cast<const T*>(bias), static_cast<T*>(y),
          per_tensor, rows, n, k, block_size, activation_block_size);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      return Status::OK();
    }
  }
  const int blocks = Blocks(SafeInt<int64_t>(rows) * n, kThreads / 32);
  IntegerMatMul<T, Bits><<<blocks, kThreads, 0, stream>>>(
      a, a_scales, b, b_scales, scale_type, static_cast<const T*>(bias), static_cast<T*>(y),
      per_tensor, rows, n, k, block_size, activation_block_size);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

template <typename T>
__global__ void EmptyOutput(T* y, const T* bias, int64_t count, int64_t n) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count;
       i += int64_t{gridDim.x} * blockDim.x) {
    y[i] = from_float<T>(__fadd_rn(0.0f, bias ? to_float(bias[i % n]) : 0.0f));
  }
}
}  // namespace

Status LaunchValidateDynamicWeightScales(const void* scales, DynamicWeightScaleType type,
                                         int64_t count, int* invalid, cudaStream_t stream) {
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(invalid, 0, sizeof(int), stream));
  ValidateScales<<<Blocks(count), kThreads, 0, stream>>>(scales, type, count, invalid);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchDynamicSignedWeights(const uint8_t* weights, int8_t* signed_weights, int64_t count,
                                  cudaStream_t stream) {
  ORT_RETURN_IF_NOT(count % 4 == 0, "Dynamic signed-weight preparation requires a multiple of four bytes.");
  DynamicSignedWeights<<<Blocks(count / 4), kThreads, 0, stream>>>(weights, signed_weights, count);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchDynamicExpandedWeights(const uint8_t* weights, int8_t* expanded, int bits, int64_t count,
                                    cudaStream_t stream) {
  ORT_RETURN_IF_NOT(count % 4 == 0, "Dynamic weight expansion requires a multiple of four codes.");
  const int blocks = Blocks(count / 4);
  if (bits == 8) {
    DynamicExpandWeights<8><<<blocks, kThreads, 0, stream>>>(weights, expanded, count);
  } else if (bits == 4) {
    DynamicExpandWeights<4><<<blocks, kThreads, 0, stream>>>(weights, expanded, count);
  } else if (bits == 2) {
    DynamicExpandWeights<2><<<blocks, kThreads, 0, stream>>>(weights, expanded, count);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported weight width.");
  }
  return CUDA_CALL(cudaGetLastError());
}

size_t DynamicTensorScaleScratchSize(int64_t count) {
  if (count <= kTensorReductionElementsPerBlock) return 0;
  return static_cast<size_t>(std::min<int64_t>(kTensorReductionMaxBlocks,
                                               count / kTensorReductionElementsPerBlock +
                                                   (count % kTensorReductionElementsPerBlock != 0)));
}

Status LaunchDynamicTensorScale(const void* a, bool bf16, int64_t count,
                                float* scale, float* partials, cudaStream_t stream) {
  const int blocks = static_cast<int>(DynamicTensorScaleScratchSize(count));
  if (blocks == 0) {
    if (bf16)
      TensorScale<<<1, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(a), count, scale);
    else
      TensorScale<<<1, kThreads, 0, stream>>>(static_cast<const half*>(a), count, scale);
  } else {
    ORT_RETURN_IF_NOT(partials, "Tensor scale reduction requires partial-maxima scratch.");
    if (bf16)
      TensorMaximumPartials<<<blocks, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(a), count, partials);
    else
      TensorMaximumPartials<<<blocks, kThreads, 0, stream>>>(static_cast<const half*>(a), count, partials);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    TensorScaleFromPartials<<<1, kThreads, 0, stream>>>(partials, blocks, scale);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

// Pick the narrowest group that keeps every thread within one, two, or four chunks.
template <typename T>
void LaunchQuantizeVector(const T* a, int64_t domains, int64_t k, int64_t block_size, int64_t groups,
                          int64_t domain_size, const float* tensor_scale, int8_t* quantized, float* scales,
                          cudaStream_t stream) {
  auto launch = [&](auto kernel, int group) {
    const int64_t per_block = kThreads / group;
    kernel<<<static_cast<int>((domains + per_block - 1) / per_block), kThreads, 0, stream>>>(
        a, domains, k, block_size, groups, tensor_scale, quantized, scales);
  };
  if (domain_size <= 32 * kVectorElements) {
    launch(QuantizeRowsVector<T, 32, 1>, 32);
  } else if (domain_size <= 64 * kVectorElements) {
    launch(QuantizeRowsVector<T, 64, 1>, 64);
  } else if (domain_size <= 128 * kVectorElements) {
    launch(QuantizeRowsVector<T, 128, 1>, 128);
  } else if (domain_size <= 256 * kVectorElements) {
    launch(QuantizeRowsVector<T, 256, 1>, 256);
  } else if (domain_size <= 256 * kVectorElements * 2) {
    launch(QuantizeRowsVector<T, 256, 2>, 256);
  } else {
    launch(QuantizeRowsVector<T, 256, kVectorMaxChunks>, 256);
  }
}

Status LaunchDynamicQuantizeRows(const void* a, bool bf16, int64_t rows, int64_t k,
                                 int64_t block_size, const float* tensor_scale,
                                 int8_t* quantized, float* scales, cudaStream_t stream) {
  const int64_t groups = k / block_size + (k % block_size != 0);
  const int64_t domain_size = std::min(block_size, k);
  const bool vector_ok = k % kVectorElements == 0 && (block_size >= k || block_size % kVectorElements == 0) &&
                         domain_size <= int64_t{kThreads} * kVectorElements * kVectorMaxChunks &&
                         rows * groups <= std::numeric_limits<int>::max() / 2 &&
                         reinterpret_cast<uintptr_t>(a) % 16 == 0 && reinterpret_cast<uintptr_t>(quantized) % 8 == 0;
  if (vector_ok) {
    const int64_t domains = rows * groups;
    if (bf16) {
      LaunchQuantizeVector(static_cast<const __nv_bfloat16*>(a), domains, k, block_size, groups, domain_size,
                           tensor_scale, quantized, scales, stream);
    } else {
      LaunchQuantizeVector(static_cast<const half*>(a), domains, k, block_size, groups, domain_size,
                           tensor_scale, quantized, scales, stream);
    }
  } else if (bf16) {
    const int blocks = Blocks(SafeInt<int64_t>(rows) * groups, 1);
    QuantizeRows<<<blocks, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(a), rows, k, block_size,
                                                  tensor_scale, quantized, scales);
  } else {
    const int blocks = Blocks(SafeInt<int64_t>(rows) * groups, 1);
    QuantizeRows<<<blocks, kThreads, 0, stream>>>(static_cast<const half*>(a), rows, k, block_size,
                                                  tensor_scale, quantized, scales);
  }
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchDynamicQuantizedMatMul(const int8_t* a, const float* a_scales,
                                    const uint8_t* b, const void* b_scales, DynamicWeightScaleType scale_type,
                                    const void* bias, void* y, bool bf16, bool per_tensor,
                                    int64_t rows, int64_t n, int64_t k,
                                    int bits, int64_t block_size, int64_t activation_block_size,
                                    int64_t weight_prepacked, int compute_capability, cudaStream_t stream) {
#define LAUNCH_DYNAMIC_MATMUL(Bits)                                                                                      \
  return bf16 ? LaunchMatMul<__nv_bfloat16, Bits>(a, a_scales, b, b_scales, scale_type, bias, y, per_tensor, rows, n, k, \
                                                  block_size, activation_block_size, weight_prepacked,                   \
                                                  compute_capability, stream)                                            \
              : LaunchMatMul<half, Bits>(a, a_scales, b, b_scales, scale_type, bias, y, per_tensor, rows, n, k,          \
                                         block_size, activation_block_size, weight_prepacked, compute_capability, stream)
  switch (bits) {
    case 8:
      LAUNCH_DYNAMIC_MATMUL(8);
    case 4:
      LAUNCH_DYNAMIC_MATMUL(4);
    case 2:
      LAUNCH_DYNAMIC_MATMUL(2);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "bits must be 2, 4, or 8.");
  }
#undef LAUNCH_DYNAMIC_MATMUL
}

Status LaunchDynamicQuantizedEmpty(void* y, const void* bias, bool bf16,
                                   int64_t count, int64_t n, cudaStream_t stream) {
  if (bf16) {
    EmptyOutput<<<Blocks(count), kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(y),
                                                        static_cast<const __nv_bfloat16*>(bias), count, n);
  } else {
    EmptyOutput<<<Blocks(count), kThreads, 0, stream>>>(static_cast<half*>(y), static_cast<const half*>(bias), count, n);
  }
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
