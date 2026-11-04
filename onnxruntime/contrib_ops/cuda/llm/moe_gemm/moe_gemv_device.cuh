// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Shared device-side machinery for the symmetric weight-only MoE GEMV fast path.
// Contains fpA_intB_gemv kernels and host launch/dispatch helper templates used by
// the MXFP4 public launchers.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// Accumulator element of the K-paired inner loop, see accumulate_column_tile below. 16-bit
// accumulation keeps the two k lanes of the vec2 apart until the epilogue; fp32 accumulation
// reduces each pair to one float right away.
template <typename Details, typename AccT>
using TileAccType =
    std::conditional_t<std::is_same_v<AccT, float>, float,
                       typename MathWrapper<typename Details::TypeDetailsA>::Type2>;

// Accumulates the K tile of one output column into `acc`.
//
// This replaces the dequantize/pack_to_vec2/mma sequence of the dense fpA_intB GEMV, which pairs
// the products across two *columns* so that one hfma2 serves both. That pairing needs the decoded
// weights shuffled into column-major register pairs, costing one prmt per weight pair -- about a
// ninth of the QMoE decode GEMV's issued instructions. Pairing along K instead needs no shuffle at
// all: the activation tile is already in k order and the converters emit k pairs contiguously
// (Mapper sends a logical pair to a physical pair, and to an even physical index, so the vec2 load
// stays aligned).
//
// Apply the group scale to each decoded weight before multiplying by the activation. Besides
// matching the original dequantize-then-mma order, this prevents an unscaled fp16 product from
// overflowing even when the final scaled product is representable. The fp32 policy converts each
// scaled pair before multiplying so BF16 products are not accumulated in BF16 first.
//
// Numerics: this is a reassociation, not a rewrite. The 16-bit policy keeps the even-k and odd-k
// partial sums in the two halves of one vec2 and only adds them together in collapse_tile_acc,
// where the previous code summed a column's K terms in one serial chain. Floating-point addition
// is not associative, so results are close but not bit-identical to that order; the per-thread
// chain is now half as long, which if anything reduces accumulated rounding error.
template <typename Details, int K, typename TileAccT, typename TypeA>
__device__ __forceinline__ void accumulate_column_tile(TileAccT& acc, void const* w, void const* act, TypeA scale) {
  using Math = MathWrapper<typename Details::TypeDetailsA>;
  using Type = typename Math::Type;
  using Type2 = typename Math::Type2;
  static_assert(K % 2 == 0);
  typename Details::LayoutDetails::Mapper mapper;

  Type2 const zero = Math::to_vec2(static_cast<Type>(0.f));
  Type2 const scale2 = Math::to_vec2(scale);
#pragma unroll
  for (int j = 0; j < K / 2; ++j) {
    Type2 const w2 = *reinterpret_cast<Type2 const*>(reinterpret_cast<Type const*>(w) + mapper(2 * j));
    Type2 const scaled_w2 = Math::fma2(w2, scale2, zero);
    Type2 const a2 = reinterpret_cast<Type2 const*>(act)[j];
    if constexpr (std::is_same_v<TileAccT, float>) {
      float2 const scaled_w_f2 = Math::to_float2(scaled_w2);
      float2 const a_f2 = Math::to_float2(a2);
      acc += scaled_w_f2.x * a_f2.x + scaled_w_f2.y * a_f2.y;
    } else {
      acc = Math::fma2(scaled_w2, a2, acc);
    }
  }
}

// Collapses the K-paired accumulators into the one-value-per-column form the epilogues expect.
template <typename Details, int CtaN, typename AccT, typename TileAccT>
__device__ __forceinline__ void collapse_tile_acc(AccT* tile_acc, TileAccT const* tile_k_acc) {
  using Math = MathWrapper<typename Details::TypeDetailsA>;
#pragma unroll
  for (int i = 0; i < CtaN; ++i) {
    if constexpr (std::is_same_v<TileAccT, float>) {
      tile_acc[i] = tile_k_acc[i];
    } else {
      float2 const p = Math::to_float2(tile_k_acc[i]);
      tile_acc[i] = static_cast<AccT>(p.x + p.y);
    }
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__global__ void moe_gemv_kernel(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                                int num_experts,
                                int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k) {
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
  using ScalesAccessT = ScalesAccess<TypeA, CtaN, Details::kInterleave>;
  GMemIterator<Mandatory, typename ScalesAccessT::TVec, ScalesAccessT::kStrided, ScalesAccessT::kContinuous, TypeA>
      scales_iterator(
          scales,
          (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
          (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  using Converter = typename ConverterWrapper<Details>::Converter;
  using TileAccT = TileAccType<Details, AccT>;
  using Math = MathWrapper<typename Details::TypeDetailsA>;

  TileAccT tile_k_acc[CtaN];
  if constexpr (std::is_same_v<TileAccT, float>) {
    fill<CtaN>(tile_k_acc, 0.f);
  } else {
    fill<CtaN>(tile_k_acc, Math::to_vec2(static_cast<typename Math::Type>(0.f)));
  }

  // load_scales() writes through a ScalesAccessT::TVec* (float4 when vectorized), and the
  // iterators/converters below write tile_a through AccessTypeA* and tile_w through uint32_t*,
  // so these arrays need the alignment of the widest access, not just of TypeA.
  alignas(alignof(typename ScalesAccessT::TVec)) TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
    load_scales<ScalesAccessT, CtaN>(scales_iterator, vec_scale, 0);
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    alignas(alignof(AccessTypeA)) TypeA tile_a[StepK];
    // Issue all CtaN weight loads before consuming any of them: interleaving a load with its own
    // decode leaves a single load in flight and makes the kernel long-scoreboard bound.
    AccessTypeW tile_w_quantized[CtaN * Details::kAccessNumW];
    if constexpr (GroupSize != 0) {
      load_scales<ScalesAccessT, CtaN>(scales_iterator, vec_scale, iter);
    }
    act_iterator.load(tile_a, iter, 0);
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized + i * Details::kAccessNumW, iter, i);
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      alignas(alignof(uint32_t)) TypeA tile_w[StepK];
      Converter::template convert<StepK>(tile_w_quantized + i * Details::kAccessNumW, tile_w);
      accumulate_column_tile<Details, StepK>(tile_k_acc[i], tile_w, tile_a, vec_scale[i]);
    }
  }

  AccT tile_acc[CtaM * CtaN];
  collapse_tile_acc<Details, CtaN>(tile_acc, tile_k_acc);
  epilogue<Details, CtaM, CtaN, Threads, EnableBias, false, AccT>(out, n, tile_acc, bias, 1.0f);
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

template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__global__ void moe_gemv_interleaved_swiglu_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int inter_size, int k,
    cutlass_kernels::ActivationParams activation_params,
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

  const int source_row = permuted_row_to_source_row ? permuted_row_to_source_row[row] % num_rows : offset_m;
  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, source_row * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  using ScalesAccessT = ScalesAccess<TypeA, CtaN, Details::kInterleave>;
  GMemIterator<Mandatory, typename ScalesAccessT::TVec, ScalesAccessT::kStrided, ScalesAccessT::kContinuous, TypeA>
      scales_iterator(
          scales,
          (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
          (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * inter_size + tile_id_n * CtaN * Details::kInterleave / 2;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  using Converter = typename ConverterWrapper<Details>::Converter;
  using TileAccT = TileAccType<Details, AccT>;
  using Math = MathWrapper<typename Details::TypeDetailsA>;

  TileAccT tile_k_acc[CtaN];
  if constexpr (std::is_same_v<TileAccT, float>) {
    fill<CtaN>(tile_k_acc, 0.f);
  } else {
    fill<CtaN>(tile_k_acc, Math::to_vec2(static_cast<typename Math::Type>(0.f)));
  }

  // See moe_gemv_kernel: these are written through wider pointer casts than TypeA.
  alignas(alignof(typename ScalesAccessT::TVec)) TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
    load_scales<ScalesAccessT, CtaN>(scales_iterator, vec_scale, 0);
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    alignas(alignof(AccessTypeA)) TypeA tile_a[StepK];
    // See moe_gemv_kernel: keep all CtaN weight loads in flight at once.
    AccessTypeW tile_w_quantized[CtaN * Details::kAccessNumW];
    if constexpr (GroupSize != 0) {
      load_scales<ScalesAccessT, CtaN>(scales_iterator, vec_scale, iter);
    }
    act_iterator.load(tile_a, iter, 0);
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized + i * Details::kAccessNumW, iter, i);
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      alignas(alignof(uint32_t)) TypeA tile_w[StepK];
      Converter::template convert<StepK>(tile_w_quantized + i * Details::kAccessNumW, tile_w);
      accumulate_column_tile<Details, StepK>(tile_k_acc[i], tile_w, tile_a, vec_scale[i]);
    }
  }

  AccT tile_acc[CtaM * CtaN];
  collapse_tile_acc<Details, CtaN>(tile_acc, tile_k_acc);
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
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows, cudaStream_t stream) {
  const int64_t n = inter_size * 2;
  const int64_t weight_expert_stride = n * k / Details::kElemsPerByteW;
  const int64_t scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, num_rows);
  } else {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, num_rows);
  }
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
  } else if (group_size == 16) {
    launch_moe_gemv<Details, CtaN, Threads, 16, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
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
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows, cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 0, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows, stream);
  } else if (group_size == 16) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 16, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows, stream);
  } else if (group_size == 32) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 32, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows, stream);
  } else if (group_size == 64) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 64, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows, stream);
  } else if (group_size == 128) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 128, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, permuted_row_to_source_row, num_rows, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
}

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
