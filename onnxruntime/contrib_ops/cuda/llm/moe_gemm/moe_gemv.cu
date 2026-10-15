// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"

#include <cuda_fp16.h>
#include <type_traits>

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"
// Include env_var_utils.h after dispatcher.h: dispatcher.h transitively pulls in provider_api.h,
// which defines SHARED_PROVIDER. That guard suppresses env_var_utils.h's own logging.h include and
// avoids redefining CREATE_MESSAGE / LOGS_CATEGORY (which would fail under -Werror).
#include "core/platform/env_var_utils.h"

namespace onnxruntime::llm::kernels::fpA_intB_gemv {

template <typename Details>
struct Int8ActivationDetails : Details {};

template <typename Details>
struct UsesInt8Activation : std::false_type {};

template <typename Details>
struct UsesInt8Activation<Int8ActivationDetails<Details>> : std::true_type {};

template <typename Details>
using DecodeWeight = std::conditional_t<UsesInt8Activation<Details>::value, int32_t,
                                      typename Details::TypeDetailsA::Type>;

template <typename Details, int StepK>
inline constexpr int DecodeWeightElements = UsesInt8Activation<Details>::value ? StepK / 4 : StepK;

template <typename Details, int StepK>
__device__ __forceinline__ void moe_pack_weights(void* destination, void* source, int column) {
  if constexpr (UsesInt8Activation<Details>::value) {
#pragma unroll
    for (int group = 0; group < StepK / 4; ++group) {
      reinterpret_cast<int32_t*>(destination)[column * (StepK / 4) + group] =
          reinterpret_cast<const int32_t*>(source)[group];
    }
  } else {
    pack_to_vec2<Details, StepK>(destination, source, column);
  }
}

template <typename Details, int StepK>
__device__ __forceinline__ void moe_decode_weights(void* weights, void* packed, void* scales) {
  if constexpr (UsesInt8Activation<Details>::value) {
    static_assert(StepK == 64 && Details::kInterleave == 8 && Details::LayoutDetails::kTileSize == 64);
    const auto* words = reinterpret_cast<const uint32_t*>(packed);
    const uint32_t even_bytes = __byte_perm(words[0], words[1], 0x6420);
    const uint32_t odd_bytes = __byte_perm(words[2], words[3], 0x6420);
    const uint32_t even_high_bytes = __byte_perm(words[0], words[1], 0x7531);
    const uint32_t odd_high_bytes = __byte_perm(words[2], words[3], 0x7531);
#pragma unroll
    for (int group = 0; group < StepK / 4; ++group) {
      const uint32_t gathered = group < 8
                                    ? (group % 2 == 0 ? even_bytes : odd_bytes)
                                    : (group % 2 == 0 ? even_high_bytes : odd_high_bytes);
      const int shift = (group / 2) % 4 * 2;
      const uint32_t codes = (gathered >> shift) & 0x03030303u;
      reinterpret_cast<int32_t*>(weights)[group] = static_cast<int32_t>(__vsub4(codes, 0x02020202u));
    }
  } else {
    dequantize<Details, 1, StepK, false, false>(weights, packed, scales, nullptr, 1.0f);
  }
}

template <typename Details, int CtaN, int StepK, typename AccT>
__device__ __forceinline__ void moe_decode_dot(AccT* accumulator, void* weights, void* activation, void* scales,
                                              const int8_t* quantized_input = nullptr,
                                              const float* input_scales = nullptr) {
  if constexpr (UsesInt8Activation<Details>::value) {
    static_assert(std::is_same_v<AccT, float>);
    using Type = typename Details::TypeDetailsA::Type;
    auto* values = reinterpret_cast<Type*>(activation);
    auto* weight_values = reinterpret_cast<const int32_t*>(weights);
    auto* weight_scales = reinterpret_cast<Type*>(scales);
    constexpr int QuantTile = 16;
    static_assert(StepK % QuantTile == 0);
    float scaled_dot[CtaN];
#pragma unroll
    for (int column = 0; column < CtaN; ++column) {
      scaled_dot[column] = 0.f;
    }
#pragma unroll
    for (int offset = 0; offset < StepK; offset += QuantTile) {
      float maximum = 0.f;
      if (quantized_input == nullptr) {
#pragma unroll
        for (int element = 0; element < QuantTile; ++element) {
          maximum = fmaxf(maximum, fabsf(static_cast<float>(values[offset + element])));
        }
      }
      float scale = quantized_input != nullptr ? input_scales[offset / QuantTile] : maximum / 127.f;
      float inverse_scale = maximum > 0.f ? 127.f / maximum : 0.f;
      int packed_activations[QuantTile / 4];
#pragma unroll
      for (int group = 0; group < QuantTile / 4; ++group) {
        if (quantized_input != nullptr) {
          packed_activations[group] = reinterpret_cast<const int32_t*>(quantized_input)[offset / 4 + group];
        } else {
          unsigned int packed_value = 0;
#pragma unroll
          for (int lane = 0; lane < 4; ++lane) {
            int quantized = __float2int_rn(static_cast<float>(values[offset + group * 4 + lane]) * inverse_scale);
            quantized = max(-127, min(127, quantized));
            packed_value |= static_cast<unsigned int>(static_cast<uint8_t>(quantized)) << (lane * 8);
          }
          packed_activations[group] = static_cast<int>(packed_value);
        }
      }
#pragma unroll
      for (int column = 0; column < CtaN; ++column) {
        int dot = 0;
#pragma unroll
        for (int group = 0; group < QuantTile / 4; ++group) {
          dot = __dp4a(packed_activations[group],
                       weight_values[column * (StepK / 4) + offset / 4 + group], dot);
        }
        scaled_dot[column] += static_cast<float>(dot) * scale;
      }
    }
#pragma unroll
    for (int column = 0; column < CtaN; ++column) {
      accumulator[column] += scaled_dot[column] * static_cast<float>(weight_scales[column]);
    }
  } else {
    mma<Details, 1, CtaN, StepK, AccT>(accumulator, weights, activation);
  }
}

}

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// Resolves which activation row an expanded row reads from.
//
// By default the activations have already been replicated by expandInputRows, so expanded row i
// reads act row i. When `permuted_row_to_source_row` is non-null the caller skipped that expansion
// and handed us the *unexpanded* activations instead, so we follow the same mapping
// finalizeMoeRouting uses: permuted row -> unpermuted row in [0, num_rows * top_k), then modulo
// num_rows to get the token. This turns the replication into a re-read of one row, which the L2
// serves, and removes a kernel launch from the decode path.
template <int CtaM>
__device__ __forceinline__ int act_source_row(const int* permuted_row_to_source_row, int num_rows,
                                              int row, int offset_m) {
  if (permuted_row_to_source_row == nullptr) {
    return offset_m;
  }
  static_assert(CtaM == 1, "source-row indirection assumes one expanded row per block");
  return permuted_row_to_source_row[row] % num_rows;
}

// MoE batched GEMV kernel. One thread-block (CtaM = 1 row) processes a single
// expanded row; the row's expert determines the weight/scale/bias base pointers.
// Mirrors the dense fpA_intB_gemv `kernel<>` body for GroupSize=0 (per-channel),
// EnableActScale=false, EnableZero=false, ApplyAlphaInAdvance=false.
template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA, bool Prequantized = false>
__global__ void moe_gemv_kernel(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                                int num_experts,
                                int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k,
                                const int8_t* quantized_input = nullptr, const float* input_scales = nullptr,
                                const int* input_rows = nullptr) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  const int row = blockIdx.x;

  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
  // Fallback path for prologues that have not materialized the row-to-expert map.
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;
  if constexpr (EnableBias) {
    bias += static_cast<int64_t>(expert) * n;
  }

  const int origin_k = k, interleaved_k = k * Details::kInterleave;

  const int tile_id_m = row, tile_id_n = blockIdx.y, tid = threadIdx.x;
  const int offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  const int real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  const int real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  AccT tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<AccT>(0.f));

  TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, 0, i);
    }
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    TypeA tile_a[StepK];
    DecodeWeight<Details> tile_w[DecodeWeightElements<Details, StepK>],
      tile_w_pack2[CtaN * DecodeWeightElements<Details, StepK>];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
    if constexpr (GroupSize != 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, iter, i);
      }
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      moe_decode_weights<Details, StepK>(tile_w, tile_w_quantized, vec_scale + i);
      moe_pack_weights<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      if constexpr (Prequantized) {
        const int input_row = input_rows != nullptr ? input_rows[row] : offset_m;
        const int input_offset = real_offset_k + iter * (CtaK / Details::kInterleave);
        moe_decode_dot<Details, CtaN, StepK, AccT>(
            tile_acc + i * CtaN, tile_w_pack2, nullptr, vec_scale,
            quantized_input + input_row * origin_k + input_offset,
            input_scales + input_row * (origin_k / 16) + input_offset / 16);
      } else {
        act_iterator.load(tile_a, iter, i);
        moe_decode_dot<Details, CtaN, StepK, AccT>(tile_acc + i * CtaN, tile_w_pack2, tile_a, vec_scale);
      }
    }
  }
  epilogue<Details, CtaM, CtaN, Threads, EnableBias, false, AccT>(out, n, tile_acc, bias, 1.0f);
#endif
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)) || !defined(__CUDA_ARCH__)
// Epilogue for the fused-finalize FC2 GEMV. Reduces one expert's column tile exactly like
// `epilogue()` but, instead of storing it, scales it by that expert's routing weight and
// accumulates it into `out_acc` (shared, CtaN * kInterleave floats) so the block can sum all
// top_k experts of a token before a single store.
template <typename Details, int CtaN, int Threads, bool EnableBias, typename AccT>
__device__ __forceinline__ void fused_finalize_accumulate(float* out_acc, void* tile_acc, void* bias,
                                                          float row_scale) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static_assert(Threads % WarpSize == 0);

  __shared__ float shmem[CtaN * Interleave * WarpNum];
  const int tid = threadIdx.x;
  const int warp_id = tid / WarpSize, lane_id = tid % WarpSize;

  // Ordering barrier. It serves two distinct purposes depending on the iteration:
  //   1. First expert: the `out_acc` zero-fill performed by the caller before entering the expert
  //      loop must be visible to all threads before any thread writes `shmem` and, further down,
  //      accumulates into `out_acc`.
  //   2. Subsequent experts: the `out_acc` reads/writes of the previous call must complete before
  //      this call overwrites `shmem` for the new expert.
  // The enclosing expert loop is block-uniform, so every thread reaches this barrier the same
  // number of times.
  __syncthreads();
#pragma unroll
  for (int n = 0; n < CtaN; ++n) {
    float v = static_cast<float>(reinterpret_cast<AccT*>(tile_acc)[n]);
    v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
    if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
      shmem[warp_id * CtaN * Interleave + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
    }
  }
  __syncthreads();
  for (int ii = tid; ii < CtaN * Interleave; ii += Threads) {
    float val = 0.f, v_bias = 0.f;
    if constexpr (EnableBias) {
      v_bias = static_cast<float>(reinterpret_cast<Type*>(bias)[ii]);
    }
#pragma unroll
    for (int jj = 0; jj < WarpNum; ++jj) {
      val += shmem[jj * CtaN * Interleave + ii];
    }
    out_acc[ii] += row_scale * (val + v_bias);
  }
}
#endif  // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)) || !defined(__CUDA_ARCH__)

// FC2 GEMV with the MoE finalize (top_k weighted reduction) fused into the epilogue.
//
// The unfused sequence writes each of the top_k expanded rows to a scratch buffer and then runs
// `finalizeMoeRouting` as a separate kernel. For batch-1 decode that reduction is launched with a
// single block, so its launch and drain latency dominates the trivial amount of work it does.
// Here one block owns all top_k partials of one token for one column tile: it loops over the
// token's experts, accumulates `scale * (dot + bias)` in shared memory, and stores the reduced row
// straight to the MoE output. That removes a kernel launch plus the scratch round trip to DRAM.
//
// grid = (num_rows, n / (CtaN * kInterleave)). Because the top_k loop moved inside the block, the
// grid is top_k times narrower than the unfused GEMV at the same CtaN; callers compensate with a
// smaller CtaN so the block count still covers the machine.
//
// Layout notes (matching finalizeMoeRoutingKernelLauncher):
//   unpermuted_row_to_permuted_row is indexed [k_idx * num_rows + token]
//   final_scales is indexed          [token * experts_per_token + k_idx]
// Only valid for single-EP / single-TP (every selected expert is local and bias applies here).
template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__global__ void moe_gemv_fused_finalize_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int* unpermuted_row_to_permuted_row, const int* permuted_row_to_expert, const float* final_scales,
    int num_experts, int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k,
    int num_rows, int experts_per_token) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static constexpr int Interleave = Details::kInterleave;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Interleave) % GroupSize == 0);
  }

  const int token = blockIdx.x, tile_id_n = blockIdx.y, tid = threadIdx.x;
  const int origin_k = k, interleaved_k = k * Interleave;
  const int interleaved_offset_n = tile_id_n * CtaN;
  const int real_offset_n =
      interleaved_offset_n * Interleave + ((tid * StepK / Details::LayoutDetails::kTileSize) % Interleave);
  const int real_offset_k =
      (tid * StepK / (Interleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  __shared__ float out_acc[CtaN * Interleave];
  for (int ii = tid; ii < CtaN * Interleave; ii += Threads) {
    out_acc[ii] = 0.f;
  }

  // Keep this a real loop: unrolling would replicate the whole K-loop body per expert.
#pragma unroll 1
  for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
    const int permuted_row = unpermuted_row_to_permuted_row[k_idx * num_rows + token];
    const int expert = permuted_row >= 0 ? permuted_row_to_expert[permuted_row] : -1;
    if (expert < 0 || expert >= num_experts) {
      continue;
    }

    uint8_t* expert_weight = weight + expert * weight_expert_stride;
    TypeA* expert_scales = scales + static_cast<int64_t>(expert) * scale_expert_stride;
    TypeA* expert_bias = nullptr;
    if constexpr (EnableBias) {
      expert_bias = bias + static_cast<int64_t>(expert) * n + tile_id_n * CtaN * Interleave;
    }

    GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
        act, permuted_row * origin_k + real_offset_k, CtaK / Interleave, origin_k);
    GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
        expert_weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
        CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
    GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
        expert_scales, (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
        (GroupSize != 0 ? CtaK / Interleave / GroupSize * n : 0), Interleave);

    AccT tile_acc[CtaN];
    fill<CtaN>(tile_acc, static_cast<AccT>(0.f));

    TypeA vec_scale[CtaN];
    if constexpr (GroupSize == 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, 0, i);
      }
    }

    for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
      TypeA tile_a[StepK];
        DecodeWeight<Details> tile_w[DecodeWeightElements<Details, StepK>],
          tile_w_pack2[CtaN * DecodeWeightElements<Details, StepK>];
      uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
      if constexpr (GroupSize != 0) {
#pragma unroll
        for (int i = 0; i < CtaN; ++i) {
          scales_iterator.load(vec_scale + i, iter, i);
        }
      }
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        weight_iterator.load(tile_w_quantized, iter, i);
        moe_decode_weights<Details, StepK>(tile_w, tile_w_quantized, vec_scale + i);
        moe_pack_weights<Details, StepK>(tile_w_pack2, tile_w, i);
      }
      act_iterator.load(tile_a, iter, 0);
      moe_decode_dot<Details, CtaN, StepK, AccT>(tile_acc, tile_w_pack2, tile_a, vec_scale);
    }

    const float row_scale = final_scales != nullptr ? final_scales[token * experts_per_token + k_idx] : 1.0f;
    fused_finalize_accumulate<Details, CtaN, Threads, EnableBias, AccT>(out_acc, tile_acc, expert_bias, row_scale);
  }

  __syncthreads();
  Type* out_row = reinterpret_cast<Type*>(out) + static_cast<int64_t>(token) * n + tile_id_n * CtaN * Interleave;
  for (int ii = tid; ii < CtaN * Interleave; ii += Threads) {
    out_row[ii] = static_cast<Type>(out_acc[ii]);
  }
#endif
}

template <typename Details, int CtaM, int CtaN, int Threads, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__device__ __forceinline__ void swiglu_epilogue(void* out, void* tile_acc, void* bias,
                                                cutlass_kernels::ActivationParams activation_params) {
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static constexpr int RawCols = CtaN * Interleave;
  static_assert(CtaM == 1);
  static_assert(RawCols % 2 == 0);
  static_assert(Threads % WarpSize == 0);

  __shared__ float shmem[CtaM * CtaN * Interleave * WarpNum];
  int tid = threadIdx.x;
  int warp_id = tid / WarpSize, lane_id = tid % WarpSize;
#pragma unroll
  for (int n = 0; n < CtaN; ++n) {
    float v = static_cast<float>(reinterpret_cast<AccT*>(tile_acc)[n]);
    v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
    if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
      shmem[warp_id * RawCols + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
    }
  }
  __syncthreads();

#pragma unroll
  for (int pair = tid; pair < RawCols / 2; pair += Threads) {
    const int gate_idx = pair * 2;
    const int linear_idx = gate_idx + 1;
    float gate = 0.f;
    float linear = 0.f;
#pragma unroll
    for (int warp = 0; warp < WarpNum; ++warp) {
      gate += shmem[warp * RawCols + gate_idx];
      linear += shmem[warp * RawCols + linear_idx];
    }
    if constexpr (EnableBias) {
      gate += static_cast<float>(reinterpret_cast<TypeA*>(bias)[gate_idx]);
      linear += static_cast<float>(reinterpret_cast<TypeA*>(bias)[linear_idx]);
    }
    if (isfinite(activation_params.limit)) {
      gate = fminf(gate, activation_params.limit);
      linear = fminf(fmaxf(linear, -activation_params.limit), activation_params.limit);
    }
    linear += activation_params.beta;
    const float sigmoid = 1.0f / (1.0f + expf(-activation_params.alpha * gate));
    reinterpret_cast<TypeA*>(out)[pair] = static_cast<TypeA>(gate * sigmoid * linear);
  }
}

template <typename Details, int CtaM, int CtaN, int Threads, typename AccT>
__device__ __forceinline__ void partial_epilogue(float* partial_out, void* tile_acc) {
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static_assert(CtaM == 1);
  static_assert(Threads % WarpSize == 0);

  __shared__ float shmem[CtaM * CtaN * Interleave * WarpNum];
  int tid = threadIdx.x;
  int warp_id = tid / WarpSize;
  int lane_id = tid % WarpSize;
#pragma unroll
  for (int n = 0; n < CtaN; ++n) {
    float v = static_cast<float>(reinterpret_cast<AccT*>(tile_acc)[n]);
    v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
    if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
      shmem[warp_id * CtaN * Interleave + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = tid; col < CtaN * Interleave; col += Threads) {
    float val = 0.f;
#pragma unroll
    for (int warp = 0; warp < WarpNum; ++warp) {
      val += shmem[warp * CtaN * Interleave + col];
    }
    partial_out[col] = val;
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = float>
__global__ void moe_gemv_splitk_partials_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, float* partials,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k, int64_t expanded_num_rows,
    const int* permuted_row_to_source_row, int num_rows) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  const int row = blockIdx.x;
  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;

  const int origin_k = k;
  const int interleaved_k = k * Details::kInterleave;
  const int tile_id_m = row;
  const int tile_id_n = blockIdx.y;
  const int split_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int offset_m = tile_id_m * CtaM;
  const int interleaved_offset_n = tile_id_n * CtaN;
  const int real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  const int real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) *
          Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, act_source_row<CtaM>(permuted_row_to_source_row, num_rows, row, offset_m) * origin_k + real_offset_k,
      CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales, (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  partials += (static_cast<int64_t>(split_id) * expanded_num_rows + offset_m) * n +
              tile_id_n * CtaN * Details::kInterleave;

  AccT tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<AccT>(0.f));

  TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, 0, i);
    }
  }

  const int num_iters = (interleaved_k + CtaK - 1) / CtaK;
  for (int iter = split_id; iter < num_iters; iter += SplitK) {
    const int idx_k = iter * CtaK + tid * StepK;
    if (idx_k >= interleaved_k) {
      continue;
    }
    TypeA tile_a[StepK];
    DecodeWeight<Details> tile_w[DecodeWeightElements<Details, StepK>];
    DecodeWeight<Details> tile_w_pack2[CtaN * DecodeWeightElements<Details, StepK>];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
    if constexpr (GroupSize != 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, iter, i);
      }
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      moe_decode_weights<Details, StepK>(tile_w, tile_w_quantized, vec_scale + i);
      moe_pack_weights<Details, StepK>(tile_w_pack2, tile_w, i);
    }
    act_iterator.load(tile_a, iter, 0);
    moe_decode_dot<Details, CtaN, StepK, AccT>(tile_acc, tile_w_pack2, tile_a, vec_scale);
  }
  partial_epilogue<Details, CtaM, CtaN, Threads, AccT>(partials, tile_acc);
#endif
}

template <typename Details, int Threads, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void moe_gemv_splitk_reduce_swiglu_kernel(
    const float* partials, TypeA* bias, TypeA* out,
    const int* permuted_row_to_expert, int num_experts, const int64_t* expert_first_token_offset,
    int inter_size, int split_k, int64_t expanded_num_rows,
    cutlass_kernels::ActivationParams activation_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  const int row = blockIdx.x;
  const int col = blockIdx.y * Threads + threadIdx.x;
  if (col >= inter_size) {
    return;
  }

  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const float* alpha = activation_params.swiglu_alpha;
  const float* beta = activation_params.swiglu_beta;
  const float* limit = activation_params.swiglu_limit;
  const float act_alpha = alpha ? alpha[expert] : activation_params.alpha;
  const float act_beta = beta ? beta[expert] : activation_params.beta;
  const float act_limit = limit ? limit[expert] : activation_params.limit;

  const int n = inter_size * 2;
  const int gate_idx = col * 2;
  const int linear_idx = gate_idx + 1;
  const int64_t row_base = static_cast<int64_t>(row) * n;
  const int64_t split_stride = expanded_num_rows * n;
  float gate = 0.f;
  float linear = 0.f;
  for (int split = 0; split < split_k; ++split) {
    const int64_t base = static_cast<int64_t>(split) * split_stride + row_base;
    gate += partials[base + gate_idx];
    linear += partials[base + linear_idx];
  }

  if constexpr (EnableBias) {
    bias += static_cast<int64_t>(expert) * n;
    gate += static_cast<float>(bias[gate_idx]);
    linear += static_cast<float>(bias[linear_idx]);
  }
  if (isfinite(act_limit)) {
    gate = fminf(gate, act_limit);
    linear = fminf(fmaxf(linear, -act_limit), act_limit);
  }
  linear += act_beta;
  const float sigmoid = 1.0f / (1.0f + expf(-act_alpha * gate));
  out[static_cast<int64_t>(row) * inter_size + col] = static_cast<TypeA>(gate * sigmoid * linear);
#endif
}

template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA, bool Prequantized = false>
__global__ void moe_gemv_interleaved_swiglu_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int inter_size, int k,
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows,
    const int8_t* quantized_input = nullptr, const float* input_scales = nullptr) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  const int row = blockIdx.x;

  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const float* alpha = activation_params.swiglu_alpha;
  const float* beta = activation_params.swiglu_beta;
  const float* limit = activation_params.swiglu_limit;
  activation_params.alpha = alpha ? alpha[expert] : activation_params.alpha;
  activation_params.beta = beta ? beta[expert] : activation_params.beta;
  activation_params.limit = limit ? limit[expert] : activation_params.limit;

  const int n = inter_size * 2;
  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;
  if constexpr (EnableBias) {
    bias += static_cast<int64_t>(expert) * n;
  }

  const int origin_k = k, interleaved_k = k * Details::kInterleave;

  const int tile_id_m = row, tile_id_n = blockIdx.y, tid = threadIdx.x;
  const int offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  const int real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  const int real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, act_source_row<CtaM>(permuted_row_to_source_row, num_rows, row, offset_m) * origin_k + real_offset_k,
      CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * inter_size + tile_id_n * CtaN * Details::kInterleave / 2;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  AccT tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<AccT>(0.f));

  TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, 0, i);
    }
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    TypeA tile_a[StepK];
    DecodeWeight<Details> tile_w[DecodeWeightElements<Details, StepK>],
      tile_w_pack2[CtaN * DecodeWeightElements<Details, StepK>];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
    if constexpr (GroupSize != 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, iter, i);
      }
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      moe_decode_weights<Details, StepK>(tile_w, tile_w_quantized, vec_scale + i);
      moe_pack_weights<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      if constexpr (Prequantized) {
        const int source = act_source_row<CtaM>(permuted_row_to_source_row, num_rows, row, offset_m);
        const int input_offset = real_offset_k + iter * (CtaK / Details::kInterleave);
        moe_decode_dot<Details, CtaN, StepK, AccT>(
            tile_acc + i * CtaN, tile_w_pack2, nullptr, vec_scale,
            quantized_input + source * origin_k + input_offset,
            input_scales + source * (origin_k / 16) + input_offset / 16);
      } else {
        act_iterator.load(tile_a, iter, i);
        moe_decode_dot<Details, CtaN, StepK, AccT>(tile_acc + i * CtaN, tile_w_pack2, tile_a, vec_scale);
      }
    }
  }
  swiglu_epilogue<Details, CtaM, CtaN, Threads, EnableBias, TypeA, AccT>(out, tile_acc, bias, activation_params);
#endif
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA>
static void launch_moe_gemv(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                            const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                            int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k,
                            cudaStream_t stream) {
  const int64_t weight_expert_stride = n * k / Details::kElemsPerByteW;
  const int64_t scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k));
  } else {
    moe_gemv_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k));
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA>
static void launch_moe_gemv_interleaved_swiglu(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k,
    cutlass_kernels::ActivationParams activation_params, const int* permuted_row_to_source_row,
    int64_t num_rows, cudaStream_t stream) {
  const int64_t n = inter_size * 2;
  const int64_t weight_expert_stride = n * k / Details::kElemsPerByteW;
  const int64_t scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, static_cast<int>(num_rows));
  } else {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, static_cast<int>(num_rows));
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK, typename TypeA>
static void launch_moe_gemv_splitk_twopass_swiglu(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k,
    cutlass_kernels::ActivationParams activation_params, const int* permuted_row_to_source_row,
    int64_t num_rows, float* partials, cudaStream_t stream) {
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  const int64_t n = inter_size * 2;
  const int64_t interleaved_k = k * Details::kInterleave;
  const int num_iters = static_cast<int>((interleaved_k + CtaK - 1) / CtaK);
  if (partials == nullptr || num_iters < 2) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, GroupSize, TypeA, float>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows, stream);
    return;
  }

  const int64_t weight_expert_stride = n * k / Details::kElemsPerByteW;
  const int64_t scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid1(static_cast<unsigned>(expanded_num_rows),
             static_cast<unsigned>(n / (CtaN * Details::kInterleave)), SplitK);
  dim3 block1(Threads);
  moe_gemv_splitk_partials_kernel<Details, CtaN, Threads, GroupSize, SplitK, TypeA, float>
      <<<grid1, block1, 0, stream>>>(
          act, weight, scales, partials, expert_first_token_offset, permuted_row_to_expert, num_experts,
          weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), expanded_num_rows,
          permuted_row_to_source_row, static_cast<int>(num_rows));

  static constexpr int kReduceThreads = 256;
  dim3 grid2(static_cast<unsigned>(expanded_num_rows),
             static_cast<unsigned>((inter_size + kReduceThreads - 1) / kReduceThreads));
  dim3 block2(kReduceThreads);
  if (bias != nullptr) {
    moe_gemv_splitk_reduce_swiglu_kernel<Details, kReduceThreads, true><<<grid2, block2, 0, stream>>>(
        partials, bias, out, permuted_row_to_expert, num_experts, expert_first_token_offset,
        static_cast<int>(inter_size), SplitK, expanded_num_rows, activation_params);
  } else {
    moe_gemv_splitk_reduce_swiglu_kernel<Details, kReduceThreads, false><<<grid2, block2, 0, stream>>>(
        partials, bias, out, permuted_row_to_expert, num_experts, expert_first_token_offset,
        static_cast<int>(inter_size), SplitK, expanded_num_rows, activation_params);
  }
}

template <typename Details, int CtaN, int Threads, int SplitK, typename TypeA>
static void dispatch_moe_gemv_splitk_twopass_swiglu_group_size(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size,
    cutlass_kernels::ActivationParams activation_params, const int* permuted_row_to_source_row,
    int64_t num_rows, float* partials, cudaStream_t stream) {
#define LAUNCH_MOE_GEMV_SPLITK_SWIGLU(GROUP_SIZE)                                                     \
  launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, GROUP_SIZE, SplitK, TypeA>(           \
      act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts, \
      expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows,      \
      partials, stream)
  if (group_size <= 0) {
    LAUNCH_MOE_GEMV_SPLITK_SWIGLU(0);
  } else if (group_size == 32) {
    LAUNCH_MOE_GEMV_SPLITK_SWIGLU(32);
  } else if (group_size == 64) {
    LAUNCH_MOE_GEMV_SPLITK_SWIGLU(64);
  } else if (group_size == 128) {
    LAUNCH_MOE_GEMV_SPLITK_SWIGLU(128);
  } else {
    ORT_THROW("unsupported MoE GEMV split-K group_size: ", group_size);
  }
#undef LAUNCH_MOE_GEMV_SPLITK_SWIGLU
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA>
static void dispatch_moe_gemv_group_size(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                         const int64_t* expert_first_token_offset,
                                         const int* permuted_row_to_expert, int num_experts,
                                         int64_t expanded_num_rows, int64_t n, int64_t k,
                                         int group_size, cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv<Details, CtaN, Threads, 0, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                            permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else if (group_size == 32) {
    launch_moe_gemv<Details, CtaN, Threads, 32, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                             permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else if (group_size == 64) {
    launch_moe_gemv<Details, CtaN, Threads, 64, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                             permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else if (group_size == 128) {
    launch_moe_gemv<Details, CtaN, Threads, 128, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                              permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA>
static void dispatch_moe_gemv_interleaved_swiglu_group_size(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size,
    cutlass_kernels::ActivationParams activation_params, const int* permuted_row_to_source_row,
    int64_t num_rows, cudaStream_t stream) {
#define LAUNCH_MOE_GEMV_INTERLEAVED_SWIGLU(GROUP_SIZE)                                                \
  launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, GROUP_SIZE, TypeA, AccT>(                \
      act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts, \
      expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows, stream)
  if (group_size <= 0) {
    LAUNCH_MOE_GEMV_INTERLEAVED_SWIGLU(0);
  } else if (group_size == 32) {
    LAUNCH_MOE_GEMV_INTERLEAVED_SWIGLU(32);
  } else if (group_size == 64) {
    LAUNCH_MOE_GEMV_INTERLEAVED_SWIGLU(64);
  } else if (group_size == 128) {
    LAUNCH_MOE_GEMV_INTERLEAVED_SWIGLU(128);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
#undef LAUNCH_MOE_GEMV_INTERLEAVED_SWIGLU
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA>
static void launch_moe_gemv_fused_finalize(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int* unpermuted_row_to_permuted_row, const int* permuted_row_to_expert, const float* final_scales,
    int num_experts, int64_t num_rows, int64_t experts_per_token, int64_t n, int64_t k, cudaStream_t stream) {
  const int64_t weight_expert_stride = n * k / Details::kElemsPerByteW;
  const int64_t scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_fused_finalize_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, unpermuted_row_to_permuted_row, permuted_row_to_expert, final_scales,
        num_experts, weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(num_rows), static_cast<int>(experts_per_token));
  } else {
    moe_gemv_fused_finalize_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, unpermuted_row_to_permuted_row, permuted_row_to_expert, final_scales,
        num_experts, weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(num_rows), static_cast<int>(experts_per_token));
  }
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA>
static void dispatch_moe_gemv_fused_finalize_group_size(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int* unpermuted_row_to_permuted_row, const int* permuted_row_to_expert, const float* final_scales,
    int num_experts, int64_t num_rows, int64_t experts_per_token, int64_t n, int64_t k, int group_size,
    cudaStream_t stream) {
#define LAUNCH_MOE_GEMV_FUSED_FINALIZE(GROUP_SIZE)                                            \
  launch_moe_gemv_fused_finalize<Details, CtaN, Threads, GROUP_SIZE, TypeA, AccT>(            \
      act, weight, scales, bias, out, unpermuted_row_to_permuted_row, permuted_row_to_expert, \
      final_scales, num_experts, num_rows, experts_per_token, n, k, stream)
  if (group_size <= 0) {
    LAUNCH_MOE_GEMV_FUSED_FINALIZE(0);
  } else if (group_size == 32) {
    LAUNCH_MOE_GEMV_FUSED_FINALIZE(32);
  } else if (group_size == 64) {
    LAUNCH_MOE_GEMV_FUSED_FINALIZE(64);
  } else if (group_size == 128) {
    LAUNCH_MOE_GEMV_FUSED_FINALIZE(128);
  } else {
    ORT_THROW("unsupported MoE GEMV fused-finalize group_size: ", group_size);
  }
#undef LAUNCH_MOE_GEMV_FUSED_FINALIZE
}

}  // namespace fpA_intB_gemv

namespace moe_gemv {

namespace fiv = onnxruntime::llm::kernels::fpA_intB_gemv;

// CtaN/Threads match the dense per-channel (EnableZero=false) m-tile config.
static constexpr int kCtaN = 8;
static constexpr int kThreads = 128;
// int4 ColumnMajorInterleave (Sm80) tile width along N.
static constexpr int kTileSizeK = 64;

constexpr int WeightInterleave(int weight_bits) {
  return 128 * 8 / (kTileSizeK * weight_bits);
}

// The fused-finalize FC2 GEMV moves the top_k loop inside the block, so at kCtaN it would launch
// top_k times fewer blocks than the unfused kernel and lose memory-level parallelism. A narrower
// column tile restores the block count; CtaN=2 also measured fastest for the FC2 shape on sm_120.
static constexpr int kFusedFinalizeCtaN = 2;

// Maps a runtime accumulator-precision choice to a compile-time type tag so the
// host launcher can select the fp32- or 16-bit-accumulation kernel instantiation.
template <typename T>
struct TypeTag {
  using type = T;
};

// Opt-in: accumulate the GEMV inner product in fp32 instead of the default fp16
// for fp16 activations. bf16 always accumulates in fp32 because 16-bit bf16
// accumulation is too lossy.
inline bool MoeGemvUseFp32Accum() {
  // Parsed once via ORT's environment helper (consistent parsing/thread-safety across platforms).
  const static bool enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_MOE_GEMV_FP32_ACCUM", 0) == 1;
  return enabled;
}

namespace {
// Shared shape checks for every MoE GEMV variant. cta_n is the launching variant's CtaN, which sets
// the n-tiling requirement: the plain GEMVs use kCtaN, the fused-finalize kernel uses the smaller
// kFusedFinalizeCtaN and therefore accepts n values the plain kernels reject.
bool is_moe_gemv_shape_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k,
                                 int weight_bits, int group_size, int cta_n) {
  if (sm < 80) {
    return false;
  }
  if (weight_bits != 2 && weight_bits != 4 && weight_bits != 8) {
    return false;
  }
  // group_size <= 0 selects the per-column (per-channel) path; block-wise scales must be 32, 64, or 128.
  if (group_size > 0 && group_size != 32 && group_size != 64 && group_size != 128) {
    return false;
  }
  // Keep the first block-wise GEMV implementation on complete K blocks.
  if (group_size > 0 && k % group_size != 0) {
    return false;
  }
  if (expanded_num_rows <= 0 || expanded_num_rows > kMaxProfiledExpandedRows) {
    return false;
  }
  if (n < kMinProfiledProblemDim || k < kMinProfiledProblemDim) {
    return false;
  }
  if (expanded_num_rows > kMaxProfiledExpandedRowsForSmallProblemDim &&
      (n < kMinProfiledProblemDimForExpandedRowsAbove4 || k < kMinProfiledProblemDimForExpandedRowsAbove4)) {
    return false;
  }
  // n must tile evenly; k must tile evenly into StepK along interleaved-K.
  const int interleave = WeightInterleave(weight_bits);
  if (n % (cta_n * interleave) != 0) {
    return false;
  }
  const int64_t interleaved_k = k * interleave;
  const int step_k = 128 / weight_bits;
  if (interleaved_k % step_k != 0) {
    return false;
  }

  // The interleaved kernel reads K in whole tiles of kTileSizeK (64). Each group of participating
  // threads covers one kTileSizeK-wide K tile via sub-offsets {0, kTileSizeK/2, ...}; the per-thread
  // activation iterator unconditionally loads StepK elements at real_offset_k. When k is not a multiple
  // of kTileSizeK the final partial tile makes the upper-half threads read past the valid k range of the
  // activation row (e.g. k=32 reads act[..32..63]), yielding garbage/NaN. Require complete K tiles.
  if (k % kTileSizeK != 0) {
    return false;
  }

  return true;
}
}  // namespace

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k,
                           int weight_bits, int group_size) {
  const int cta_n = weight_bits == 2 ? kCtaN / 2 : kCtaN;
  return is_moe_gemv_shape_supported(sm, expanded_num_rows, n, k, weight_bits, group_size, cta_n);
}

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k) {
  return is_moe_gemv_supported(sm, expanded_num_rows, n, k, 4, 0);
}

bool is_moe_gemv_fused_finalize_supported(int sm, int64_t num_rows, int64_t experts_per_token,
                                          int64_t expanded_num_rows, int64_t n, int64_t k,
                                          int weight_bits, int group_size) {
  // The fused-finalize kernel is launched with kFusedFinalizeCtaN, so it only needs n to tile by
  // that (smaller) CtaN rather than the kCtaN the plain GEMVs require.
  if (!is_moe_gemv_shape_supported(sm, expanded_num_rows, n, k, weight_bits, group_size,
                                   kFusedFinalizeCtaN)) {
    return false;
  }
  if (num_rows <= 0 || experts_per_token <= 0) {
    return false;
  }
  // The fused reduction assumes every token contributes exactly experts_per_token permuted rows,
  // which is what expandInputRows produces when no tokens are dropped.
  if (num_rows * experts_per_token != expanded_num_rows) {
    return false;
  }
  return true;
}

template <typename T, typename WeightType>
struct DetailsForTAndWeight;

template <typename WeightType>
struct CtaNForWeight : std::integral_constant<int, kCtaN> {};

template <>
struct CtaNForWeight<cutlass::uint2b_t> : std::integral_constant<int, kCtaN / 2> {};

template <>
struct DetailsForTAndWeight<half, cutlass::uint2b_t> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int2DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
  static constexpr int kWeightBits = 2;
};

template <>
struct DetailsForTAndWeight<half, cutlass::uint4b_t> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
  static constexpr int kWeightBits = 4;
};

template <>
struct DetailsForTAndWeight<half, uint8_t> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int8DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
  static constexpr int kWeightBits = 8;
};

#ifdef ENABLE_BF16
template <>
struct DetailsForTAndWeight<__nv_bfloat16, cutlass::uint2b_t> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int2DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
  static constexpr int kWeightBits = 2;
};

template <>
struct DetailsForTAndWeight<__nv_bfloat16, cutlass::uint4b_t> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
  static constexpr int kWeightBits = 4;
};

template <>
struct DetailsForTAndWeight<__nv_bfloat16, uint8_t> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int8DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
  static constexpr int kWeightBits = 8;
};
#endif

__global__ void quantize_moe_input16(const half* input, int8_t* output, float* scales, int tiles) {
  const int tile = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile >= tiles) {
    return;
  }
  float maximum = 0.f;
#pragma unroll
  for (int element = 0; element < 16; ++element) {
    maximum = fmaxf(maximum, fabsf(__half2float(input[tile * 16 + element])));
  }
  const float inverse_scale = maximum > 0.f ? 127.f / maximum : 0.f;
  scales[tile] = maximum / 127.f;
#pragma unroll
  for (int element = 0; element < 16; ++element) {
    const int quantized = __float2int_rn(__half2float(input[tile * 16 + element]) * inverse_scale);
    output[tile * 16 + element] = static_cast<int8_t>(max(-127, min(127, quantized)));
  }
}

void launch_moe_quantize_input16_experiment(const half* input, int8_t* output, float* scales,
                                           int input_rows, int depth, cudaStream_t stream) {
  ORT_ENFORCE(input_rows > 0 && depth > 0 && depth % 16 == 0);
  const int tiles = input_rows * (depth / 16);
  quantize_moe_input16<<<(tiles + 127) / 128, 128, 0, stream>>>(input, output, scales, tiles);
}

void launch_moe_prequantized_int2_experiment(const int8_t* input, const float* input_scales,
                                            const uint8_t* weight, const half* scales, half* output,
                                            const int* experts, const int* input_rows, int num_experts,
                                            int expanded_rows, int columns, int depth, int group_size,
                                            cudaStream_t stream) {
  using Details = fiv::Int8ActivationDetails<typename DetailsForTAndWeight<half, cutlass::uint2b_t>::Details>;
  constexpr int ColumnsPerTile = CtaNForWeight<cutlass::uint2b_t>::value;
  ORT_ENFORCE(expanded_rows > 0 && depth % kTileSizeK == 0 &&
              columns % (ColumnsPerTile * Details::kInterleave) == 0);
  ORT_ENFORCE(group_size == 64 || group_size == 128);
  const dim3 grid(expanded_rows, columns / (ColumnsPerTile * Details::kInterleave));
  auto launch = [&](auto group_tag) {
    constexpr int Group = decltype(group_tag)::value;
    fiv::moe_gemv_kernel<Details, ColumnsPerTile, kThreads, Group, false, half, float, true>
        <<<grid, kThreads, 0, stream>>>(
            nullptr, const_cast<uint8_t*>(weight), const_cast<half*>(scales), nullptr, output,
            nullptr, experts, num_experts, static_cast<int64_t>(columns) * depth / 4,
            static_cast<int64_t>(depth / Group) * columns, columns, depth, input, input_scales, input_rows);
  };
  if (group_size == 64) {
    launch(std::integral_constant<int, 64>{});
  } else {
    launch(std::integral_constant<int, 128>{});
  }
}

void launch_moe_prequantized_int2_swiglu_experiment(
    const int8_t* input, const float* input_scales, const uint8_t* weight, const half* scales,
    half* output, const int* experts, const int* source_rows, int num_experts,
    int expanded_rows, int num_rows, int inter_size, int depth, int group_size,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  using Details = fiv::Int8ActivationDetails<typename DetailsForTAndWeight<half, cutlass::uint2b_t>::Details>;
  constexpr int ColumnsPerTile = CtaNForWeight<cutlass::uint2b_t>::value;
  const int columns = inter_size * 2;
  ORT_ENFORCE(expanded_rows > 0 && num_rows > 0 && depth > 0 && depth % kTileSizeK == 0 &&
              columns > 0 && columns % (ColumnsPerTile * Details::kInterleave) == 0);
  ORT_ENFORCE(group_size == 64 || group_size == 128);
  const dim3 grid(expanded_rows, columns / (ColumnsPerTile * Details::kInterleave));
  auto launch = [&](auto group_tag) {
    constexpr int Group = decltype(group_tag)::value;
    fiv::moe_gemv_interleaved_swiglu_kernel<Details, ColumnsPerTile, kThreads, Group, false, half, float, true>
        <<<grid, kThreads, 0, stream>>>(
            nullptr, const_cast<uint8_t*>(weight), const_cast<half*>(scales), nullptr, output,
            nullptr, experts, num_experts, static_cast<int64_t>(columns) * depth / 4,
            static_cast<int64_t>(depth / Group) * columns, inter_size, depth,
            activation_params, source_rows, num_rows, input, input_scales);
  };
  if (group_size == 64) {
    launch(std::integral_constant<int, 64>{});
  } else {
    launch(std::integral_constant<int, 128>{});
  }
}

template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric(const T* act, const WeightType* weight, const T* scales, const T* bias, T* out,
                                   const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                                   int num_experts,
                                   int64_t expanded_num_rows, int64_t n, int64_t k, int group_size, int sm,
                                   cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForTAndWeight<T, WeightType>::Details;
  using TypeA = typename DetailsForTAndWeight<T, WeightType>::TypeA;
  // Accumulate fp16 activations in fp16 by default. ORT_MOE_GEMV_FP32_ACCUM=1
  // restores the previous fp32 accumulation path; bf16 always uses fp32.
  const bool use_fp32_accum = std::is_same_v<WeightType, cutlass::uint2b_t> ||
                              !std::is_same_v<T, half> || MoeGemvUseFp32Accum();
  auto launch = [&](auto acc_tag, auto details_tag) {
    using AccT = typename decltype(acc_tag)::type;
    using SelectedDetails = typename decltype(details_tag)::type;
    fiv::dispatch_moe_gemv_group_size<SelectedDetails, CtaNForWeight<WeightType>::value, kThreads, TypeA, AccT>(
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(act)),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(weight)),
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(scales)),
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(bias)),
        reinterpret_cast<TypeA*>(out),
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
  };
  if constexpr (std::is_same_v<WeightType, cutlass::uint2b_t>) {
    if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_INT2_INT8_ACTIVATIONS", 0) == 1) {
      launch(TypeTag<float>{}, TypeTag<fiv::Int8ActivationDetails<Details>>{});
      return;
    }
  }
  if (use_fp32_accum) {
    launch(TypeTag<float>{}, TypeTag<Details>{});
  } else {
    launch(TypeTag<TypeA>{}, TypeTag<Details>{});
  }
}

template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric_fused_finalize(
    const T* act, const WeightType* weight, const T* scales, const T* bias, T* out,
    const int* unpermuted_row_to_permuted_row, const int* permuted_row_to_expert, const float* final_scales,
    int num_experts, int64_t num_rows, int64_t experts_per_token, int64_t n, int64_t k, int group_size, int sm,
    cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForTAndWeight<T, WeightType>::Details;
  using TypeA = typename DetailsForTAndWeight<T, WeightType>::TypeA;
  // Accumulation policy matches launch_moe_gemv_int_symmetric.
  const bool use_fp32_accum = std::is_same_v<WeightType, cutlass::uint2b_t> ||
                              !std::is_same_v<T, half> || MoeGemvUseFp32Accum();
  auto launch = [&](auto acc_tag, auto details_tag) {
    using AccT = typename decltype(acc_tag)::type;
    using SelectedDetails = typename decltype(details_tag)::type;
    fiv::dispatch_moe_gemv_fused_finalize_group_size<SelectedDetails, kFusedFinalizeCtaN, kThreads, TypeA, AccT>(
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(act)),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(weight)),
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(scales)),
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(bias)),
        reinterpret_cast<TypeA*>(out),
        unpermuted_row_to_permuted_row, permuted_row_to_expert, final_scales, num_experts, num_rows,
        experts_per_token, n, k, group_size, stream);
  };
  if constexpr (std::is_same_v<WeightType, cutlass::uint2b_t>) {
    if (ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_INT2_INT8_ACTIVATIONS", 0) == 1) {
      launch(TypeTag<float>{}, TypeTag<fiv::Int8ActivationDetails<Details>>{});
      return;
    }
  }
  if (use_fp32_accum) {
    launch(TypeTag<float>{}, TypeTag<Details>{});
  } else {
    launch(TypeTag<TypeA>{}, TypeTag<Details>{});
  }
}

template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric_interleaved_swiglu(
    const T* act, const WeightType* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size, int sm,
    cutlass_kernels::ActivationParams activation_params, const int* permuted_row_to_source_row,
    int64_t num_rows, float* splitk_partials, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForTAndWeight<T, WeightType>::Details;
  using TypeA = typename DetailsForTAndWeight<T, WeightType>::TypeA;
  // Accumulation policy matches launch_moe_gemv_int_symmetric.
  const bool use_fp32_accum = std::is_same_v<WeightType, cutlass::uint2b_t> ||
                              !std::is_same_v<T, half> || MoeGemvUseFp32Accum();
  const bool use_int8_activations = std::is_same_v<WeightType, cutlass::uint2b_t> &&
                                  ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_INT2_INT8_ACTIVATIONS", 0) == 1;
  // The split-K2 two-pass path always reduces FP32 partials, so it is only valid under fp32
  // accumulation. When ORT_MOE_GEMV_FP16_ACCUM=1 requests 16-bit accumulation, fall back to the
  // single-kernel path below so that env knob continues to behave as documented.
  if (splitk_partials != nullptr && use_fp32_accum && !use_int8_activations) {
    if constexpr (std::is_same_v<T, half>) {
      fiv::dispatch_moe_gemv_splitk_twopass_swiglu_group_size<Details, CtaNForWeight<WeightType>::value, kThreads, 2, TypeA>(
          const_cast<TypeA*>(reinterpret_cast<const TypeA*>(act)),
          const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(weight)),
          const_cast<TypeA*>(reinterpret_cast<const TypeA*>(scales)),
          const_cast<TypeA*>(reinterpret_cast<const TypeA*>(bias)),
          reinterpret_cast<TypeA*>(out),
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
          activation_params, permuted_row_to_source_row, num_rows, splitk_partials, stream);
      return;
    }
  }

  auto launch = [&](auto acc_tag, auto details_tag) {
    using AccT = typename decltype(acc_tag)::type;
    using SelectedDetails = typename decltype(details_tag)::type;
    if constexpr (std::is_same_v<T, half> && std::is_same_v<WeightType, cutlass::uint2b_t> &&
                  std::is_same_v<SelectedDetails, Details>) {
        if (sm == 80 && num_rows == 1 && expanded_num_rows == 8 && inter_size == 768 && k == 2048 &&
          bias == nullptr && (group_size == 64 || group_size == 128) &&
          ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_INT2_FP16_FC1_CTA8", 0) == 1) {
        fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<SelectedDetails, 8, kThreads, TypeA, AccT>(
            const_cast<TypeA*>(reinterpret_cast<const TypeA*>(act)),
            const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(weight)),
            const_cast<TypeA*>(reinterpret_cast<const TypeA*>(scales)),
            const_cast<TypeA*>(reinterpret_cast<const TypeA*>(bias)),
            reinterpret_cast<TypeA*>(out),
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
            activation_params, permuted_row_to_source_row, num_rows, stream);
        return;
      }
    }
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<SelectedDetails, CtaNForWeight<WeightType>::value, kThreads, TypeA, AccT>(
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(act)),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(weight)),
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(scales)),
        const_cast<TypeA*>(reinterpret_cast<const TypeA*>(bias)),
        reinterpret_cast<TypeA*>(out),
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
        activation_params, permuted_row_to_source_row, num_rows, stream);
  };
  if constexpr (std::is_same_v<WeightType, cutlass::uint2b_t>) {
    if (use_int8_activations) {
      launch(TypeTag<float>{}, TypeTag<fiv::Int8ActivationDetails<Details>>{});
      return;
    }
  }
  if (use_fp32_accum) {
    launch(TypeTag<float>{}, TypeTag<Details>{});
  } else {
    launch(TypeTag<TypeA>{}, TypeTag<Details>{});
  }
}

template <typename T>
void launch_moe_gemv_int4_per_channel(const T* act, const uint8_t* weight, const T* scales, const T* bias, T* out,
                                      const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                                      int num_experts,
                                      int64_t expanded_num_rows, int64_t n, int64_t k, int sm, cudaStream_t stream) {
  launch_moe_gemv_int_symmetric<T, cutlass::uint4b_t>(
      act, reinterpret_cast<const cutlass::uint4b_t*>(weight), scales, bias, out, expert_first_token_offset,
      permuted_row_to_expert, num_experts, expanded_num_rows, n, k, 0, sm, stream);
}

template <typename T>
void launch_moe_gemv_int4_per_channel_interleaved_swiglu(
    const T* act, const uint8_t* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int sm,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  launch_moe_gemv_int_symmetric_interleaved_swiglu<T, cutlass::uint4b_t>(
      act, reinterpret_cast<const cutlass::uint4b_t*>(weight), scales, bias, out, expert_first_token_offset,
      permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, 0, sm, activation_params,
      /*permuted_row_to_source_row=*/nullptr, /*num_rows=*/expanded_num_rows, nullptr, stream);
}

template void launch_moe_gemv_int_symmetric<half, cutlass::uint4b_t>(
    const half*, const cutlass::uint4b_t*, const half*, const half*, half*, const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric<half, cutlass::uint2b_t>(
    const half*, const cutlass::uint2b_t*, const half*, const half*, half*, const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric<half, uint8_t>(
    const half*, const uint8_t*, const half*, const half*, half*, const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_fused_finalize<half, cutlass::uint4b_t>(
    const half*, const cutlass::uint4b_t*, const half*, const half*, half*, const int*, const int*, const float*,
    int, int64_t, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_fused_finalize<half, uint8_t>(
    const half*, const uint8_t*, const half*, const half*, half*, const int*, const int*, const float*,
    int, int64_t, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<half, cutlass::uint4b_t>(
    const half*, const cutlass::uint4b_t*, const half*, const half*, half*, const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, const int*, int64_t, float*, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<half, cutlass::uint2b_t>(
    const half*, const cutlass::uint2b_t*, const half*, const half*, half*, const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, const int*, int64_t, float*, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<half, uint8_t>(
    const half*, const uint8_t*, const half*, const half*, half*, const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, const int*, int64_t, float*, cudaStream_t);

template void launch_moe_gemv_int4_per_channel<half>(const half*, const uint8_t*, const half*, const half*, half*,
                                                     const int64_t*, const int*, int, int64_t, int64_t, int64_t,
                                                     int, cudaStream_t);
template void launch_moe_gemv_int4_per_channel_interleaved_swiglu<half>(
    const half*, const uint8_t*, const half*, const half*, half*, const int64_t*, const int*, int, int64_t,
    int64_t, int64_t, int, cutlass_kernels::ActivationParams, cudaStream_t);

#ifdef ENABLE_BF16
template void launch_moe_gemv_int_symmetric<__nv_bfloat16, cutlass::uint2b_t>(
    const __nv_bfloat16*, const cutlass::uint2b_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric<__nv_bfloat16, cutlass::uint4b_t>(
    const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric<__nv_bfloat16, uint8_t>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_fused_finalize<__nv_bfloat16, cutlass::uint4b_t>(
    const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int*, const int*, const float*, int, int64_t, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_fused_finalize<__nv_bfloat16, uint8_t>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int*, const int*, const float*, int, int64_t, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<__nv_bfloat16, cutlass::uint4b_t>(
    const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    const int*, int64_t, float*, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<__nv_bfloat16, cutlass::uint2b_t>(
    const __nv_bfloat16*, const cutlass::uint2b_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    const int*, int64_t, float*, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<__nv_bfloat16, uint8_t>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    const int*, int64_t, float*, cudaStream_t);

template void launch_moe_gemv_int4_per_channel<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, cudaStream_t);
template void launch_moe_gemv_int4_per_channel_interleaved_swiglu<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, cutlass_kernels::ActivationParams,
    cudaStream_t);
#endif
}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
