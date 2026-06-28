// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Shared device-side machinery for the symmetric weight-only MoE GEMV fast path.
// Contains the fpA_intB_gemv kernels and host launch/dispatch helper templates that
// are reused by both the INT (moe_gemv.cu) and MXFP4 (moe_gemv_fp4.cu) public launchers.
// Extracted into a header so the two translation units share one definition.

#pragma once

#include <cuda_fp16.h>
#include <type_traits>

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// MoE batched GEMV kernel. One thread-block (CtaM = 1 row) processes a single
// expanded row; the row's expert determines the weight/scale/bias base pointers.
// Mirrors the dense fpA_intB_gemv `kernel<>` body for GroupSize=0 (per-channel),
// EnableActScale=false, EnableZero=false, ApplyAlphaInAdvance=false.
template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__global__ void moe_gemv_kernel(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
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

  int const row = blockIdx.x;

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

  int const origin_k = k, interleaved_k = k * Details::kInterleave;

  int const tile_id_m = row, tile_id_n = blockIdx.y, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k =
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
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
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
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      act_iterator.load(tile_a, iter, i);
      mma<Details, 1, CtaN, StepK, AccT>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
    }
  }
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
    int const gate_idx = pair * 2;
    int const linear_idx = gate_idx + 1;
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
    float const sigmoid = 1.0f / (1.0f + expf(-activation_params.alpha * gate));
    reinterpret_cast<TypeA*>(out)[pair] = static_cast<TypeA>(gate * sigmoid * linear);
  }
}

template <typename Details, int CtaM, int CtaN, int Threads, typename AccT, typename PartialT>
__device__ __forceinline__ void partial_epilogue(PartialT* partial_out, void* tile_acc) {
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
    partial_out[col] = static_cast<PartialT>(val);
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = float, typename PartialT = AccT>
__global__ void moe_gemv_splitk_partials_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, PartialT* partials,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k, int64_t expanded_num_rows) {
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

  int const row = blockIdx.x;
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

  int const origin_k = k;
  int const interleaved_k = k * Details::kInterleave;
  int const tile_id_m = row;
  int const tile_id_n = blockIdx.y;
  int const split_id = blockIdx.z;
  int const tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM;
  int const interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) *
          Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
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

  int const num_iters = (interleaved_k + CtaK - 1) / CtaK;
  for (int iter = split_id; iter < num_iters; iter += SplitK) {
    int const idx_k = iter * CtaK + tid * StepK;
    if (idx_k >= interleaved_k) {
      continue;
    }
    TypeA tile_a[StepK];
    TypeA tile_w[StepK];
    TypeA tile_w_pack2[CtaN * StepK];
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
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
    act_iterator.load(tile_a, iter, 0);
    mma<Details, 1, CtaN, StepK, AccT>(tile_acc, tile_w_pack2, tile_a);
  }
  partial_epilogue<Details, CtaM, CtaN, Threads, AccT, PartialT>(partials, tile_acc);
#endif
}

template <typename Details, int Threads, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename PartialT = TypeA>
__global__ void moe_gemv_splitk_reduce_swiglu_kernel(
    PartialT const* partials, TypeA* bias, TypeA* out,
    int const* permuted_row_to_expert, int num_experts, int64_t const* expert_first_token_offset,
    int inter_size, int split_k, int64_t expanded_num_rows,
    cutlass_kernels::ActivationParams activation_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  int const row = blockIdx.x;
  int const col = blockIdx.y * Threads + threadIdx.x;
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

  float const* alpha = activation_params.swiglu_alpha;
  float const* beta = activation_params.swiglu_beta;
  float const* limit = activation_params.swiglu_limit;
  float const act_alpha = alpha ? alpha[expert] : activation_params.alpha;
  float const act_beta = beta ? beta[expert] : activation_params.beta;
  float const act_limit = limit ? limit[expert] : activation_params.limit;

  int const n = inter_size * 2;
  int const gate_idx = col * 2;
  int const linear_idx = gate_idx + 1;
  int64_t const row_base = static_cast<int64_t>(row) * n;
  int64_t const split_stride = expanded_num_rows * n;
  float gate = 0.f;
  float linear = 0.f;
  for (int split = 0; split < split_k; ++split) {
    int64_t const base = static_cast<int64_t>(split) * split_stride + row_base;
    gate += static_cast<float>(partials[base + gate_idx]);
    linear += static_cast<float>(partials[base + linear_idx]);
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
  float const sigmoid = 1.0f / (1.0f + expf(-act_alpha * gate));
  out[static_cast<int64_t>(row) * inter_size + col] = static_cast<TypeA>(gate * sigmoid * linear);
#endif
}

template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__global__ void moe_gemv_interleaved_swiglu_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int inter_size, int k,
    cutlass_kernels::ActivationParams activation_params) {
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

  int const row = blockIdx.x;

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

  float const* alpha = activation_params.swiglu_alpha;
  float const* beta = activation_params.swiglu_beta;
  float const* limit = activation_params.swiglu_limit;
  activation_params.alpha = alpha ? alpha[expert] : activation_params.alpha;
  activation_params.beta = beta ? beta[expert] : activation_params.beta;
  activation_params.limit = limit ? limit[expert] : activation_params.limit;

  int const n = inter_size * 2;
  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;
  if constexpr (EnableBias) {
    bias += static_cast<int64_t>(expert) * n;
  }

  int const origin_k = k, interleaved_k = k * Details::kInterleave;

  int const tile_id_m = row, tile_id_n = blockIdx.y, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k =
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
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
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
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      act_iterator.load(tile_a, iter, i);
      mma<Details, 1, CtaN, StepK, AccT>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
    }
  }
  swiglu_epilogue<Details, CtaM, CtaN, Threads, EnableBias, TypeA, AccT>(out, tile_acc, bias, activation_params);
#endif
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA>
static void launch_moe_gemv(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                            int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                            int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k,
                            cudaStream_t stream) {
  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  int64_t const scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
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
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  int64_t const n = inter_size * 2;
  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  int64_t const scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params);
  } else {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params);
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK, typename TypeA, typename AccT,
          typename PartialT>
static void launch_moe_gemv_splitk_twopass_swiglu(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k,
    cutlass_kernels::ActivationParams activation_params, PartialT* partials, cudaStream_t stream) {
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  int64_t const n = inter_size * 2;
  int64_t const interleaved_k = k * Details::kInterleave;
  int const num_iters = static_cast<int>((interleaved_k + CtaK - 1) / CtaK);
  if (partials == nullptr || num_iters < 2) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, GroupSize, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
    return;
  }

  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  int64_t const scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid1(static_cast<unsigned>(expanded_num_rows),
             static_cast<unsigned>(n / (CtaN * Details::kInterleave)), SplitK);
  dim3 block1(Threads);
  moe_gemv_splitk_partials_kernel<Details, CtaN, Threads, GroupSize, SplitK, TypeA, AccT, PartialT>
      <<<grid1, block1, 0, stream>>>(
          act, weight, scales, partials, expert_first_token_offset, permuted_row_to_expert, num_experts,
          weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), expanded_num_rows);

  static constexpr int kReduceThreads = 256;
  dim3 grid2(static_cast<unsigned>(expanded_num_rows),
             static_cast<unsigned>((inter_size + kReduceThreads - 1) / kReduceThreads));
  dim3 block2(kReduceThreads);
  if (bias != nullptr) {
    moe_gemv_splitk_reduce_swiglu_kernel<Details, kReduceThreads, true, TypeA, PartialT><<<grid2, block2, 0, stream>>>(
        partials, bias, out, permuted_row_to_expert, num_experts, expert_first_token_offset,
        static_cast<int>(inter_size), SplitK, expanded_num_rows, activation_params);
  } else {
    moe_gemv_splitk_reduce_swiglu_kernel<Details, kReduceThreads, false, TypeA, PartialT><<<grid2, block2, 0, stream>>>(
        partials, bias, out, permuted_row_to_expert, num_experts, expert_first_token_offset,
        static_cast<int>(inter_size), SplitK, expanded_num_rows, activation_params);
  }
}

template <typename Details, int CtaN, int Threads, int SplitK, typename TypeA, typename AccT, typename PartialT>
static void dispatch_moe_gemv_splitk_twopass_swiglu_group_size(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size,
    cutlass_kernels::ActivationParams activation_params, PartialT* partials, cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 0, SplitK, TypeA, AccT, PartialT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, partials, stream);
  } else if (group_size == 32) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 32, SplitK, TypeA, AccT, PartialT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, partials, stream);
  } else if (group_size == 64) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 64, SplitK, TypeA, AccT, PartialT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, partials, stream);
  } else if (group_size == 128) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 128, SplitK, TypeA, AccT, PartialT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, partials, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV split-K group_size: ", group_size);
  }
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA>
static void dispatch_moe_gemv_group_size(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                         int64_t const* expert_first_token_offset,
                                         int const* permuted_row_to_expert, int num_experts,
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
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 0, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 32) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 32, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 64) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 64, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 128) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 128, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
}

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
