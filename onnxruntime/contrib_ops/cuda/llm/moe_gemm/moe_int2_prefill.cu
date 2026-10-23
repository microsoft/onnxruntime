// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_int2_prefill.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_int2_cache.h"
#include "cutlass/epilogue/thread/activation.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_int2.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_activation_kernels.cuh"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/details.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

#include <algorithm>
#include <limits>
#include <mma.h>
#include <cuda_pipeline.h>
#include <type_traits>
#include "cutlass/float8.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
namespace {

constexpr int kNvfp4TileK = 64;
constexpr int kNvfp4TileM = 64;
constexpr int kNvfp4TileN = 128;
constexpr int kNvfp4DownColumns = 1024;
constexpr int kNvfp4TileStride = kNvfp4TileK + 8;
constexpr int kNvfp4AElements = kNvfp4TileM * kNvfp4TileStride;
constexpr int kNvfp4PackedTileBytes = kNvfp4TileN * kNvfp4TileK / 2;
constexpr int kNvfp4PackedOffset = (2 * kNvfp4TileM + kNvfp4TileN) * kNvfp4TileStride * sizeof(half);
constexpr int kNvfp4SharedBytes = kNvfp4PackedOffset + 2 * kNvfp4PackedTileBytes;
static_assert(kNvfp4SharedBytes >= kNvfp4TileM * kNvfp4TileN * sizeof(float));

template <typename T, bool GatherInput>
__device__ void LoadNvfp4Stage(T* a, uint8_t* packed, const T* input, const uint8_t* weights,
                               int64_t first_row, int64_t end_row, int expert,
                               int first_column, int columns, int reduction, int k,
                               const int* source_rows) {
#if __CUDA_ARCH__ >= 800
  constexpr int vectors = kNvfp4TileK / 8;
  for (int i = threadIdx.x; i < kNvfp4TileM * vectors; i += blockDim.x) {
    const int64_t row = first_row + i / vectors;
    const int kk = (i % vectors) * 8;
    if (row < end_row) {
      const int64_t source_row = GatherInput ? source_rows[i / vectors] : row;
      __pipeline_memcpy_async(a + (i / vectors) * kNvfp4TileStride + kk,
                              input + source_row * reduction + k + kk, sizeof(uint4));
    } else {
      *reinterpret_cast<uint4*>(a + (i / vectors) * kNvfp4TileStride + kk) = uint4{};
    }
  }
  for (int i = threadIdx.x; i < kNvfp4TileN * vectors; i += blockDim.x) {
    const int col = first_column + i / vectors;
    const int kk = k + (i % vectors) * 8;
    if (col < columns) {
      const int64_t index = (static_cast<int64_t>(expert) * columns + col) * (reduction / 2) + kk / 2;
      __pipeline_memcpy_async(packed + i * sizeof(uint32_t), weights + index, sizeof(uint32_t));
    } else {
      reinterpret_cast<uint32_t*>(packed)[i] = 0;
    }
  }
  __pipeline_commit();
#endif
}

__global__ void BuildNvfp4TileOffsets(const int64_t* offsets, int* tiles, int experts) {
  if (threadIdx.x == 0) {
    tiles[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
      tiles[expert + 1] = tiles[expert] +
                          static_cast<int>((offsets[expert + 1] - offsets[expert] + kNvfp4TileM - 1) / kNvfp4TileM);
    }
  }
}

template <typename T, bool FuseSwiglu, bool GatherInput>
__global__ void Nvfp4GroupedGemm(const T* input, const uint8_t* weights,
                                 const uint8_t* scales, const float* global_scales,
                                 const int64_t* offsets, const int* tiles, T* output,
                                 int columns, int reduction, int experts,
                                 const T* bias, float alpha, float beta, float limit,
                                 int column_offset, int output_columns,
                                 const int* permuted_rows, int64_t num_rows) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  constexpr int stride = kNvfp4TileStride;
  // Input tiles share this storage with the FP32 epilogue, after the final K barrier.
  extern __shared__ __align__(32) uint8_t storage[];
  float* c = reinterpret_cast<float*>(storage);
  __shared__ int expert, row_tile;
  __shared__ int source_rows[GatherInput ? kNvfp4TileM : 1];
  const int warp = threadIdx.x / 32;
  const int m = (warp / 2) * 16;
  const int n = (warp % 2) * (kNvfp4TileN / 2);
  const int n_tiles = (output_columns - 1) / kNvfp4TileN + 1;
  const int64_t total = static_cast<int64_t>(tiles[experts]) * n_tiles;
  for (int64_t tile = blockIdx.x; tile < total; tile += gridDim.x) {
    if (threadIdx.x == 0) {
      const int target = static_cast<int>(tile / n_tiles);
      int lo = 0, hi = experts;
      while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (tiles[mid] <= target)
          lo = mid;
        else
          hi = mid;
      }
      expert = lo;
      row_tile = target - tiles[lo];
    }
    __syncthreads();
    const int64_t first_row = offsets[expert] + static_cast<int64_t>(row_tile) * kNvfp4TileM;
    const int64_t end_row = offsets[expert + 1];
    const int first_column = column_offset + static_cast<int>(tile % n_tiles) * kNvfp4TileN;
    const float global_scale = global_scales[expert];
    if constexpr (GatherInput) {
      if (threadIdx.x < kNvfp4TileM) {
        const int64_t row = first_row + threadIdx.x;
        source_rows[threadIdx.x] = row < end_row ? permuted_rows[row] % num_rows : 0;
      }
      __syncthreads();
    }
    constexpr int fragments = kNvfp4TileN / 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[fragments];
    for (int j = 0; j < fragments; ++j) wmma::fill_fragment(acc[j], 0.0f);
    LoadNvfp4Stage<T, GatherInput>(reinterpret_cast<T*>(storage), storage + kNvfp4PackedOffset,
                                   input, weights, first_row, end_row, expert, first_column, columns, reduction, 0, source_rows);
    for (int k = 0; k < reduction; k += kNvfp4TileK) {
      const int buffer = (k / kNvfp4TileK) % 2;
      T* a = reinterpret_cast<T*>(storage) + buffer * kNvfp4AElements;
      T* b = reinterpret_cast<T*>(storage) + 2 * kNvfp4AElements;
      const uint32_t* packed_tile = reinterpret_cast<const uint32_t*>(
          storage + kNvfp4PackedOffset + buffer * kNvfp4PackedTileBytes);
      __pipeline_wait_prior(0);
      __syncthreads();
      // Only packed weights and input rows need double buffering; the dequantized B tile is reused.
      if (k + kNvfp4TileK < reduction) {
        T* next_a = reinterpret_cast<T*>(storage) + (1 - buffer) * kNvfp4AElements;
        uint8_t* next_packed = storage + kNvfp4PackedOffset + (1 - buffer) * kNvfp4PackedTileBytes;
        LoadNvfp4Stage<T, GatherInput>(next_a, next_packed,
                                       input, weights, first_row, end_row, expert, first_column, columns, reduction, k + kNvfp4TileK,
                                       source_rows);
      }
      constexpr int vectors = kNvfp4TileK / 8;
      for (int i = threadIdx.x; i < kNvfp4TileN * vectors; i += blockDim.x) {
        const int col = first_column + i / vectors;
        const int kk = k + (i % vectors) * 8;
        alignas(16) T values[8]{};
        if (col >= columns) {
          *reinterpret_cast<uint4*>(b + (i / vectors) * stride + (i % vectors) * 8) =
              *reinterpret_cast<const uint4*>(values);
          continue;
        }
        uint32_t packed = packed_tile[i];
        fpA_intB_gemv::Fp4I2FConverter<T>::template convert<8>(&packed, values);
        const int64_t scale_index = (static_cast<int64_t>(expert) * columns + col) * (reduction / 16) + kk / 16;
        const float scale = static_cast<float>(reinterpret_cast<const cutlass::float_e4m3_t*>(scales)[scale_index]) *
                            global_scale;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float left = static_cast<float>(values[j * 2]) * scale;
          const float right = static_cast<float>(values[j * 2 + 1]) * scale;
          if constexpr (std::is_same_v<T, half>) {
            reinterpret_cast<half2*>(values)[j] = __floats2half2_rn(left, right);
          } else {
            reinterpret_cast<__nv_bfloat162*>(values)[j] = __floats2bfloat162_rn(left, right);
          }
        }
        *reinterpret_cast<uint4*>(b + (i / vectors) * stride + (i % vectors) * 8) =
            *reinterpret_cast<const uint4*>(values);
      }
      __syncthreads();
      for (int step = 0; step < kNvfp4TileK; step += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af;
        wmma::load_matrix_sync(af, a + m * stride + step, stride);
        for (int j = 0; j < fragments; ++j) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> bf;
          wmma::load_matrix_sync(bf, b + (n + j * 16) * stride + step, stride);
          wmma::mma_sync(acc[j], af, bf, acc[j]);
        }
      }
    }
    // The next iteration's copy-wait barrier also protects B reuse; only the final epilogue needs this barrier.
    __syncthreads();
    for (int j = 0; j < fragments; ++j) {
      wmma::store_matrix_sync(c + m * kNvfp4TileN + n + j * 16, acc[j], kNvfp4TileN, wmma::mem_row_major);
    }
    __syncthreads();
    if constexpr (FuseSwiglu) {
      for (int i = threadIdx.x; i < kNvfp4TileM * kNvfp4TileN / 2; i += blockDim.x) {
        const int local_row = i / (kNvfp4TileN / 2);
        const int local_col = (i % (kNvfp4TileN / 2)) * 2;
        const int64_t row = first_row + local_row;
        const int col = first_column + local_col;
        if (row < end_row && col < columns) {
          // Preserve the standalone GEMM's activation-dtype rounding before bias and SwiGLU.
          float gate = static_cast<float>(T(c[local_row * kNvfp4TileN + local_col]));
          float linear = static_cast<float>(T(c[local_row * kNvfp4TileN + local_col + 1]));
          if (bias) {
            gate += static_cast<float>(bias[static_cast<int64_t>(expert) * columns + col]);
            linear += static_cast<float>(bias[static_cast<int64_t>(expert) * columns + col + 1]);
          }
          float activated;
          if (alpha != 1.0f || beta != 0.0f || isfinite(limit)) {
            if (isfinite(limit)) {
              gate = fminf(gate, limit);
              linear = fminf(fmaxf(linear, -limit), limit);
            }
            const float sigmoid = 1.0f / (1.0f + expf(-alpha * gate));
            linear += beta;
            activated = gate * sigmoid * linear;
          } else {
            activated = cutlass::epilogue::thread::SiLu<float>{}(gate)*linear;
          }
          output[row * (columns / 2) + col / 2] = T(activated);
        }
      }
    } else {
      for (int i = threadIdx.x; i < kNvfp4TileM * kNvfp4TileN; i += blockDim.x) {
        const int64_t row = first_row + i / kNvfp4TileN;
        const int col = first_column + i % kNvfp4TileN;
        if (row < end_row && col < column_offset + output_columns) {
          output[row * output_columns + col - column_offset] = T(c[i]);
        }
      }
    }
    __syncthreads();
  }
#endif
}

template <typename ElementType>
__global__ void FinalizeNvfp4Columns(const ElementType* expanded, ElementType* output,
                                     const ElementType* bias, const float* scales,
                                     const int* reverse_map, const int* experts, int64_t rows,
                                     int hidden, int columns, int column_offset, int top_k) {
  constexpr int elements = 8;
  using Values = cutlass::Array<ElementType, elements>;
  using Compute = cutlass::Array<float, elements>;
  const int64_t row = blockIdx.x;
  for (int vector = threadIdx.x; vector < columns / elements; vector += blockDim.x) {
    Compute result;
    result.fill(0);
    for (int rank = 0; rank < top_k; ++rank) {
      const int64_t route = row * top_k + rank;
      const int permuted = reverse_map[row + rank * rows];
      const auto* input = reinterpret_cast<const Values*>(expanded + static_cast<int64_t>(permuted) * columns);
      Compute value = arrayConvert<Values, Compute>(input[vector]);
      if (bias) {
        const auto* expert_bias = reinterpret_cast<const Values*>(
            bias + static_cast<int64_t>(experts[route]) * hidden + column_offset);
        value = value + arrayConvert<Values, Compute>(expert_bias[vector]);
      }
      result = result + scales[route] * value;
    }
    auto* target = reinterpret_cast<Values*>(output + row * hidden + column_offset);
    target[vector] = arrayConvert<Compute, Values>(result);
  }
}

template <typename ElementType>
__global__ void FakeInt4Activations(ElementType* values, int64_t groups) {
  __shared__ float magnitudes[64];
  for (int64_t group = blockIdx.x; group < groups; group += gridDim.x) {
    const int lane = threadIdx.x;
    const int64_t index = group * 64 + lane;
    const float value = static_cast<float>(values[index]);
    magnitudes[lane] = fabsf(value);
    __syncthreads();
    for (int stride = 32; stride > 0; stride /= 2) {
      if (lane < stride) {
        magnitudes[lane] = fmaxf(magnitudes[lane], magnitudes[lane + stride]);
      }
      __syncthreads();
    }
    const float scale = magnitudes[0] / 7.0f;
    const float quantized = scale == 0.0f ? value : nearbyintf(value / scale) * scale;
    values[index] = static_cast<ElementType>(quantized);
    __syncthreads();
  }
}

template <typename ElementType>
void SimulateInt4Activations(ElementType* values, int64_t rows, int columns, cudaStream_t stream) {
  ORT_ENFORCE(columns > 0 && columns % 64 == 0, "INT4 activation simulation requires groups of 64.");
  const int64_t groups = SafeInt<int64_t>(rows) * (columns / 64);
  const int blocks = static_cast<int>(std::min<int64_t>(groups, 65535));
  FakeInt4Activations<<<blocks, 64, 0, stream>>>(values, groups);
  CUDA_CALL_THROW(cudaGetLastError());
}

struct PrefillWorkspace {
  size_t bytes = 0;
  size_t permuted_rows;
  size_t permuted_experts;
  size_t expert_offsets;
  size_t counts;
  size_t cumulative_counts;
  size_t blocked_rows;
  size_t expanded_input;
  size_t fc1_output;
  size_t activated_output;
  size_t fc2_output;
  size_t tile_offsets;

  size_t Reserve(size_t count, size_t element_size) {
    const size_t offset = bytes;
    const size_t end = SafeInt<size_t>(bytes) + SafeInt<size_t>(count) * element_size;
    bytes = (SafeInt<size_t>(end) + 255) / 256 * 256;
    return offset;
  }

  explicit PrefillWorkspace(const Int2MoePrefillParams& params) {
    const size_t expanded = SafeInt<size_t>(params.num_rows) * params.top_k;
    const int64_t block_rows = computeNumTokensPerBlock(params.num_rows, params.num_experts);
    const size_t blocks = (params.num_rows + block_rows - 1) / block_rows;
    const size_t count_elements = SafeInt<size_t>(params.num_experts) * blocks;
    const int input_columns = params.is_nvfp4 && params.tile_nvfp4_down_output
                                  ? std::min(params.hidden_size, kNvfp4DownColumns)
                                  : params.hidden_size;
    permuted_rows = Reserve(expanded, sizeof(int));
    permuted_experts = Reserve(expanded, sizeof(int));
    expert_offsets = Reserve(SafeInt<size_t>(params.num_experts) + 1, sizeof(int64_t));
    if (params.is_nvfp4 && params.reuse_nvfp4_routing_scratch) {
      // Sorting completes before expansion; retain maps/offsets but reuse its temporary counters.
      const size_t scratch_start = bytes;
      counts = Reserve(count_elements, sizeof(int));
      cumulative_counts = Reserve(count_elements, sizeof(int));
      blocked_rows = Reserve(SafeInt<size_t>(params.num_rows) * params.num_experts, sizeof(int));
      const size_t routing_end = bytes;
      bytes = scratch_start;
      expanded_input = Reserve(SafeInt<size_t>(expanded) * input_columns, sizeof(half));
      bytes = std::max(bytes, routing_end);
    } else {
      counts = Reserve(count_elements, sizeof(int));
      cumulative_counts = Reserve(count_elements, sizeof(int));
      blocked_rows = Reserve(SafeInt<size_t>(params.num_rows) * params.num_experts, sizeof(int));
      expanded_input = Reserve(SafeInt<size_t>(expanded) * input_columns, sizeof(half));
    }
    fc1_output = params.is_nvfp4 && params.fuse_nvfp4_swiglu
                     ? 0
                     : Reserve(SafeInt<size_t>(expanded) * params.inter_size * 2, sizeof(half));
    activated_output = Reserve(SafeInt<size_t>(expanded) * params.inter_size, sizeof(half));
    fc2_output = expanded_input;
    tile_offsets = params.is_nvfp4 ? Reserve(SafeInt<size_t>(params.num_experts) + 1, sizeof(int)) : 0;
  }
};

template <typename ElementType>
void RunPackedGroupedGemm(const Int2MoePrefillParams& params,
                          const ElementType* activations, const uint8_t* weights,
                          const void* scales, const int64_t* offsets, ElementType* output,
                          int64_t num_rows, int num_columns, int reduction_size, int weight_bits,
                          const int* tile_offsets, const float* global_scales, bool fuse_swiglu = false,
                          const int* gathered_rows = nullptr, int column_offset = 0, int output_columns = 0,
                          const uint8_t* cached_weights = nullptr) {
  if (params.is_nvfp4) {
    auto kernel = gathered_rows
                      ? (fuse_swiglu ? Nvfp4GroupedGemm<ElementType, true, true> : Nvfp4GroupedGemm<ElementType, false, true>)
                      : (fuse_swiglu ? Nvfp4GroupedGemm<ElementType, true, false> : Nvfp4GroupedGemm<ElementType, false, false>);
    CUDA_CALL_THROW(cudaFuncSetAttribute(kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, kNvfp4SharedBytes));
    kernel<<<params.multiprocessor_count * 8, (kNvfp4TileM / 16) * 64, kNvfp4SharedBytes, params.stream>>>(
        activations, weights, static_cast<const uint8_t*>(scales), global_scales,
        offsets, tile_offsets, output, num_columns, reduction_size, params.num_experts,
        static_cast<const ElementType*>(params.fc1_bias), params.alpha, params.beta, params.limit,
        column_offset, output_columns ? output_columns : num_columns, gathered_rows, params.num_rows);
    CUDA_CALL_THROW(cudaGetLastError());
    return;
  }
  if (weight_bits == 2) {
    Int2GroupedGemmParamsT<ElementType> gemm;
    gemm.activations = activations;
    gemm.packed_weights = weights;
    gemm.block_scales = static_cast<const ElementType*>(scales);
    gemm.expert_row_ends = offsets + 1;
    gemm.output = output;
    gemm.num_rows = num_rows;
    gemm.num_columns = num_columns;
    gemm.reduction_size = reduction_size;
    gemm.block_size = params.block_size;
    gemm.num_experts = params.num_experts;
    gemm.sm = params.sm;
    gemm.multiprocessor_count = params.multiprocessor_count;
    gemm.stream = params.stream;
    if (cached_weights) {
      gemm.expert_slots = params.cache_metadata;
      gemm.expert_cache_weights = cached_weights;
    }
    RunInt2GroupedGemm(gemm);
    return;
  }
  ORT_ENFORCE(weight_bits == 4, "Packed INT prefill requires INT2 or INT4 weights");
  GroupedGemmInput<ElementType, cutlass::uint4b_t, ElementType, ElementType> gemm{
      activations, offsets + 1, reinterpret_cast<const cutlass::uint4b_t*>(weights), static_cast<const ElementType*>(scales), nullptr, nullptr, output, nullptr, nullptr, ActivationType::Identity, num_rows, num_columns, reduction_size, params.num_experts, params.block_size, true, false, params.stream, {}, {}};
  gemm.gemm_config = cutlass_extensions::CutlassGemmConfig(
      cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
      cutlass_extensions::SplitKStyle::NO_SPLIT_K, 1, 4);
  MoeGemmRunner<ElementType, cutlass::uint4b_t, ElementType> runner(params.sm, params.multiprocessor_count);
  runner.moeGemm(gemm, {});
}

template <typename ElementType>
void RunInt2MoePrefillImpl(const Int2MoePrefillParams& params, void* workspace) {
  const PrefillWorkspace layout(params);
  auto* storage = static_cast<char*>(workspace);
  auto* permuted_rows = reinterpret_cast<int*>(storage + layout.permuted_rows);
  auto* permuted_experts = reinterpret_cast<int*>(storage + layout.permuted_experts);
  auto* offsets = reinterpret_cast<int64_t*>(storage + layout.expert_offsets);
  auto* expanded_input = reinterpret_cast<ElementType*>(storage + layout.expanded_input);
  auto* fc1_output = reinterpret_cast<ElementType*>(storage + layout.fc1_output);
  auto* activated_output = reinterpret_cast<ElementType*>(storage + layout.activated_output);
  auto* fc2_output = reinterpret_cast<ElementType*>(storage + layout.fc2_output);
  const int64_t expanded = params.num_rows * params.top_k;
  auto* tile_offsets = reinterpret_cast<int*>(storage + layout.tile_offsets);

  threeStepBuildExpertMapsSortFirstToken(
      params.selected_experts, permuted_experts, permuted_rows, params.unpermuted_to_permuted, offsets,
      reinterpret_cast<int*>(storage + layout.counts), reinterpret_cast<int*>(storage + layout.cumulative_counts),
      reinterpret_cast<int*>(storage + layout.blocked_rows), params.num_rows, params.num_experts, params.top_k,
      0, params.stream);
  const uint8_t* cached_fc1 = nullptr;
  const uint8_t* cached_fc2 = nullptr;
  if (params.cache_capacity > 0) {
    ORT_ENFORCE(!params.is_nvfp4 && params.fc1_weight_bits == 2 && params.fc2_weight_bits == 2 &&
                    params.cache_weights && params.cache_metadata,
                "Grouped INT2 expert cache requires uniform INT2 and owned cache buffers");
    const size_t fc1_bytes = SafeInt<size_t>(params.hidden_size) * params.inter_size / 2;
    const size_t fc2_bytes = SafeInt<size_t>(params.hidden_size) * params.inter_size / 4;
    moe_gemv::PrepareInt2ExpertCache(
        params.fc1_weights, params.fc2_weights, params.cache_weights, params.cache_metadata, offsets,
        params.num_experts, params.cache_capacity, fc1_bytes, fc2_bytes, params.stream);
    cached_fc1 = params.cache_weights;
    cached_fc2 = params.cache_weights + params.cache_capacity * fc1_bytes;
  }
  if (params.is_nvfp4) {
    BuildNvfp4TileOffsets<<<1, 32, 0, params.stream>>>(offsets, tile_offsets, params.num_experts);
    CUDA_CALL_THROW(cudaGetLastError());
  }
  const QuantParams quant_params{};
  const bool tiled_down = params.is_nvfp4 && params.tile_nvfp4_down_output;
  if (!tiled_down) {
    expandInputRowsKernelLauncher<ElementType, ElementType>(
        static_cast<const ElementType*>(params.input), expanded_input, nullptr, nullptr, permuted_rows,
        params.num_rows, params.hidden_size,
        params.top_k, params.num_experts, quant_params, false, offsets, nullptr, nullptr, nullptr, params.stream);
  }

  if (params.fake_int4_activations) {
    SimulateInt4Activations(expanded_input, expanded, params.hidden_size, params.stream);
  }
  const bool fused = params.is_nvfp4 && params.fuse_nvfp4_swiglu;
  RunPackedGroupedGemm(params, tiled_down ? static_cast<const ElementType*>(params.input) : expanded_input,
                       params.fc1_weights, params.fc1_scales,
                       offsets, fused ? activated_output : fc1_output, expanded, params.inter_size * 2,
                       params.hidden_size, params.fc1_weight_bits, tile_offsets, params.fc1_global_scales, fused,
                       tiled_down ? permuted_rows : nullptr, 0, 0, cached_fc1);

  if (!fused) {
    ActivationParams activation(ActivationType::Swiglu);
    activation.alpha = params.alpha;
    activation.beta = params.beta;
    activation.limit = params.limit;
    activation.swiglu_fusion = 1;
    doActivation<ElementType, ElementType, ElementType>(
        activated_output, fc1_output, nullptr, static_cast<const ElementType*>(params.fc1_bias), true, offsets,
        params.num_experts, params.inter_size, expanded, ActivationType::Swiglu,
        quant_params, false, nullptr, params.stream, activation);
  }

  if (params.fake_int4_activations) {
    SimulateInt4Activations(activated_output, expanded, params.inter_size, params.stream);
  }
  if (tiled_down) {
    for (int column = 0; column < params.hidden_size;) {
      const int columns = std::min(kNvfp4DownColumns, params.hidden_size - column);
      RunPackedGroupedGemm(params, activated_output, params.fc2_weights, params.fc2_scales,
                           offsets, fc2_output, expanded, params.hidden_size,
                           params.inter_size, params.fc2_weight_bits, tile_offsets, params.fc2_global_scales,
                           false, nullptr, column, columns);
      FinalizeNvfp4Columns<<<params.num_rows, 128, 0, params.stream>>>(
          fc2_output, static_cast<ElementType*>(params.output), static_cast<const ElementType*>(params.fc2_bias),
          params.routing_weights, params.unpermuted_to_permuted, params.selected_experts, params.num_rows,
          params.hidden_size, columns, column, params.top_k);
      CUDA_CALL_THROW(cudaGetLastError());
      column += columns;
    }
    return;
  }
  RunPackedGroupedGemm(params, activated_output, params.fc2_weights, params.fc2_scales,
                       offsets, fc2_output, expanded, params.hidden_size,
                       params.inter_size, params.fc2_weight_bits, tile_offsets, params.fc2_global_scales,
                       false, nullptr, 0, 0, cached_fc2);
  finalizeMoeRoutingKernelLauncher<ElementType, ElementType, ElementType>(
      fc2_output, static_cast<ElementType*>(params.output), static_cast<const ElementType*>(params.fc2_bias),
      params.routing_weights, params.unpermuted_to_permuted,
      permuted_rows, params.selected_experts, offsets, params.num_rows, params.hidden_size, params.top_k,
      params.num_experts, MOEParallelismConfig{}, false, params.stream);
}

}  // namespace

size_t GetInt2MoePrefillWorkspaceSize(const Int2MoePrefillParams& params) {
  return PrefillWorkspace(params).bytes;
}

void RunInt2MoePrefill(const Int2MoePrefillParams& params, void* workspace) {
  if (params.is_nvfp4) {
    ORT_ENFORCE(!params.tile_nvfp4_down_output || params.fuse_nvfp4_swiglu,
                "Tiled NVFP4 down output requires the fused gate/up epilogue.");
    ORT_ENFORCE(params.sm >= 80 && params.multiprocessor_count > 0 &&
                    params.num_rows > 0 && params.hidden_size > 0 && params.hidden_size % 64 == 0 &&
                    params.inter_size > 0 && params.inter_size <= std::numeric_limits<int>::max() / 2 &&
                    params.inter_size % 64 == 0 && params.num_experts > 0 &&
                    params.top_k > 0 && params.top_k <= params.num_experts &&
                    SafeInt<int64_t>(params.num_rows) * params.top_k <=
                        std::numeric_limits<int>::max() - params.num_experts,
                "Packed NVFP4 prefill requires supported tensor-core shapes and routing sizes.");
    ORT_ENFORCE(workspace && params.input && params.output && params.selected_experts &&
                    params.routing_weights && params.unpermuted_to_permuted &&
                    params.fc1_weights && params.fc2_weights && params.fc1_scales && params.fc2_scales &&
                    params.fc1_global_scales && params.fc2_global_scales,
                "Packed NVFP4 prefill requires input, output, routing, weight and scale buffers.");
  }
  if (params.is_bf16) {
#ifdef ENABLE_BF16
    RunInt2MoePrefillImpl<__nv_bfloat16>(params, workspace);
#else
    ORT_THROW("BF16 INT2 prefill requires ENABLE_BF16");
#endif
    return;
  }
  RunInt2MoePrefillImpl<half>(params, workspace);
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
