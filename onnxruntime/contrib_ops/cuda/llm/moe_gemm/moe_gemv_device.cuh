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
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fp4_fast_convert.cuh"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

template <typename Details, typename AccT>
inline constexpr bool kMoeGemvFp4FastSupported =
    IsFp4Weight<typename Details::TypeDetailsW>::value && std::is_same_v<AccT, float> &&
    Details::kInterleave > 1 && Details::kStepK == 32 && Details::kAccessNumW == 1;

template <typename Details>
__host__ __device__ constexpr int fp4_fast_layout_map(int i) {
  constexpr int kGroupA = Details::LayoutDetails::kElementGroupSizeA;
  constexpr int kGroupW = Details::LayoutDetails::kElementGroupSizeW;
  constexpr int kOffsetA = Details::LayoutDetails::kGroupOffsetA;
  return i % kGroupA + (i % kOffsetA) / kGroupA * kGroupW + i / kOffsetA * kGroupA;
}

// Decodes and accumulates one packed weight word at a time.
//
// decode_word writes registers 4w..4w+3, and those hold exactly the physical elements 8w..8w+7
// (reg_of(p) = (p/8)*4 + slot_of(p%8)/2). So a word's decoded values can be consumed before the
// next word is decoded, and only 4 of the 16 decoded registers are ever live. Decoding all four
// words up front instead costs 12 extra live registers, which at 128 threads is the difference
// between 6 and 7 blocks per SM -- and this kernel is occupancy-limited by registers, not by
// shared memory or warp slots.
//
// This was expected to be a pure reordering, but it also removes work: ncu counts 4.9% fewer
// instructions for fc1 and 24.1% fewer for fc2. Holding all 16 decoded registers live across the
// dot product left ptxas short enough of registers that it was emitting real register-to-register
// copies to keep them alive, and those are gone now. Measured per launch (H200, s_q=6):
// fc1 80 -> 71 regs, 33.3% -> 38.3% occupancy, 9.3% fewer cycles; fc2 76 -> 63 regs,
// 33.4% -> 42.6% occupancy, 28.5% fewer cycles. Local ld/st stays at exactly zero, which is what
// separates this from capping registers with __launch_bounds__: that bought the same occupancy
// but paid ~300 MB per launch in spills and came out even.
//
// Numerics: the K terms are now summed word-major rather than kk-major. Floating-point addition is
// not associative so the result is close but not bit-identical to the previous order; it is the
// same 32 products summed in a permuted order, with no change in intermediate precision.
template <typename Details, typename TypeA>
__device__ __forceinline__ void fp4_fast_accumulate(float& acc, const uint8_t* quantized, TypeA scale,
                                                    const float* act_f) {
  static constexpr int kStepK = Details::kStepK;
  static constexpr bool kPair = Details::kUseInterleavedConverter;
  using Math = MathWrapper<typename Details::TypeDetailsA>;
  using Type2 = typename Math::Type2;

  const uint32_t* words = reinterpret_cast<const uint32_t*>(quantized);
  const Type2 scale2 = Math::to_vec2(scale);
  const Type2 zero = Math::to_vec2(static_cast<TypeA>(0.f));

  float sum = acc;
#pragma unroll
  for (int w = 0; w < kStepK / 8; ++w) {
    uint32_t packed[4];
    fp4_fast::decode_word<TypeA>(words[w], packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Type2 value = Math::fma2(reinterpret_cast<Type2&>(packed[j]), scale2, zero);
      packed[j] = reinterpret_cast<uint32_t&>(value);
    }
    // Both loop counters are unrolled constants, so the guard and the register index below fold
    // away: each kk survives in exactly one w iteration.
#pragma unroll
    for (int kk = 0; kk < kStepK; ++kk) {
      const int physical = fp4_fast_layout_map<Details>(kk);
      if (physical / 8 == w) {
        const uint32_t value = packed[fp4_fast::reg_of(physical, kPair) - w * 4];
        const float weight = fp4_fast::half_of(physical, kPair) ? fp4_fast::hi_to_float<TypeA>(value)
                                                                : fp4_fast::lo_to_float<TypeA>(value);
        sum += weight * act_f[kk];
      }
    }
  }
  acc = sum;
}

// CtaM-row form of the above: one decode of each packed word feeds CtaM independent activation
// rows. This is the whole point of CtaM > 1 for this kernel -- decode is ~56 of the ~120
// instructions per word and it is row-independent, so the per-row marginal cost is only the dot.
template <typename Details, int CtaM, typename TypeA>
__device__ __forceinline__ void fp4_fast_accumulate_rows(float* acc, const uint8_t* quantized, TypeA scale,
                                                         const float act_f[CtaM][Details::kStepK]) {
  static constexpr int kStepK = Details::kStepK;
  static constexpr bool kPair = Details::kUseInterleavedConverter;
  using Math = MathWrapper<typename Details::TypeDetailsA>;
  using Type2 = typename Math::Type2;

  const uint32_t* words = reinterpret_cast<const uint32_t*>(quantized);
  const Type2 scale2 = Math::to_vec2(scale);
  const Type2 zero = Math::to_vec2(static_cast<TypeA>(0.f));

  float sum[CtaM];
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
    sum[m] = acc[m];
  }
#pragma unroll
  for (int w = 0; w < kStepK / 8; ++w) {
    uint32_t packed[4];
    fp4_fast::decode_word<TypeA>(words[w], packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Type2 value = Math::fma2(reinterpret_cast<Type2&>(packed[j]), scale2, zero);
      packed[j] = reinterpret_cast<uint32_t&>(value);
    }
#pragma unroll
    for (int kk = 0; kk < kStepK; ++kk) {
      const int physical = fp4_fast_layout_map<Details>(kk);
      if (physical / 8 == w) {
        const uint32_t value = packed[fp4_fast::reg_of(physical, kPair) - w * 4];
        const float weight = fp4_fast::half_of(physical, kPair) ? fp4_fast::hi_to_float<TypeA>(value)
                                                                : fp4_fast::lo_to_float<TypeA>(value);
#pragma unroll
        for (int m = 0; m < CtaM; ++m) {
          sum[m] += weight * act_f[m][kk];
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
    acc[m] = sum[m];
  }
}

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

// True when the permuted row this block owns has a routing weight of exactly 0.0f, i.e. nothing it
// computes can survive finalizeMoeRouting. Uniform across the block (`row` is blockIdx.x derived).
template <typename Params>
__device__ __forceinline__ bool moe_gemv_row_is_zero_weight(const Params& row_skip, int row) {
  if (row_skip.unpermuted_scales == nullptr || row_skip.permuted_row_to_unpermuted_row == nullptr) {
    return false;
  }
  const int unpermuted_row = row_skip.permuted_row_to_unpermuted_row[row];
  const int slot = unpermuted_row / row_skip.num_tokens;
  const int token = unpermuted_row - slot * row_skip.num_tokens;
  // Hard zero produced by expf(-1e30 - max) in the top-k softmax, so an exact compare is correct.
  return row_skip.unpermuted_scales[token * row_skip.experts_per_token + slot] == 0.0f;
}

// CtaM is the number of permuted rows one block owns. It is a template parameter rather than a
// constant because the caller must guarantee all CtaM rows share an expert (rows are sorted by
// expert, but a naive rows-per-block split still straddles boundaries), so CtaM > 1 needs a
// launcher that builds an expert-aligned tile map.
template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA, bool Fast = false,
          int CtaM = 1>
__global__ void moe_gemv_kernel(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                                int num_experts,
                                int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k,
                                cutlass_kernels::MoeGemvRowSkipParams row_skip) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  const int row = blockIdx.x * CtaM;

  // Zero-weight expert skip: write the zeros the epilogue would have written over this block's
  // output slice and bail out before touching any expert weight/scale bytes. With CtaM > 1 the
  // rows are only skipped wholesale when every row in the tile is zero-weight; a partially zero
  // tile still has to run, and its dead rows are zeroed in the accumulator before the epilogue.
  bool row_zero[CtaM];
  bool all_zero = true;
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
    row_zero[m] = moe_gemv_row_is_zero_weight(row_skip, row + m);
    all_zero &= row_zero[m];
  }
  if (all_zero) {
    TypeA* zero_out = out + row * n + static_cast<int>(blockIdx.y) * CtaN * Details::kInterleave;
    for (int ii = static_cast<int>(threadIdx.x); ii < CtaN * Details::kInterleave; ii += Threads) {
#pragma unroll
      for (int m = 0; m < CtaM; ++m) {
        zero_out[m * n + ii] = static_cast<TypeA>(0.f);
      }
    }
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

  TileAccT tile_k_acc[CtaM * CtaN];
  if constexpr (std::is_same_v<TileAccT, float>) {
    fill<CtaM * CtaN>(tile_k_acc, 0.f);
  } else {
    fill<CtaM * CtaN>(tile_k_acc, Math::to_vec2(static_cast<typename Math::Type>(0.f)));
  }

  // load_scales() writes through a ScalesAccessT::TVec* (float4 when vectorized), and the
  // iterators/converters below write tile_a through AccessTypeA* and tile_w through uint32_t*,
  // so these arrays need the alignment of the widest access, not just of TypeA.
  alignas(alignof(typename ScalesAccessT::TVec)) TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
    load_scales<ScalesAccessT, CtaN>(scales_iterator, vec_scale, 0);
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    alignas(alignof(AccessTypeA)) TypeA tile_a[CtaM][StepK];
    if constexpr (GroupSize != 0) {
      load_scales<ScalesAccessT, CtaN>(scales_iterator, vec_scale, iter);
    }
#pragma unroll
    for (int m = 0; m < CtaM; ++m) {
      act_iterator.load(tile_a[m], iter, m);
    }
    if constexpr (Fast) {
      static_assert(kMoeGemvFp4FastSupported<Details, AccT>);
      alignas(alignof(AccessTypeW)) uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
      float act_f[CtaM][StepK];
#pragma unroll
      for (int m = 0; m < CtaM; ++m) {
#pragma unroll
        for (int kk = 0; kk < StepK; ++kk) {
          act_f[m][kk] = static_cast<float>(tile_a[m][kk]);
        }
      }
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        weight_iterator.load(tile_w_quantized, iter, i);
        float acc_i[CtaM];
#pragma unroll
        for (int m = 0; m < CtaM; ++m) {
          acc_i[m] = tile_k_acc[m * CtaN + i];
        }
        fp4_fast_accumulate_rows<Details, CtaM>(acc_i, tile_w_quantized, vec_scale[i], act_f);
#pragma unroll
        for (int m = 0; m < CtaM; ++m) {
          tile_k_acc[m * CtaN + i] = acc_i[m];
        }
      }
    } else {
      // Keep all fallback weight loads in flight before conversion to hide memory latency.
      AccessTypeW tile_w_quantized[CtaN * Details::kAccessNumW];
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        weight_iterator.load(tile_w_quantized + i * Details::kAccessNumW, iter, i);
      }
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        alignas(alignof(uint32_t)) TypeA tile_w[StepK];
        Converter::template convert<StepK>(tile_w_quantized + i * Details::kAccessNumW, tile_w);
#pragma unroll
        for (int m = 0; m < CtaM; ++m) {
          accumulate_column_tile<Details, StepK>(tile_k_acc[m * CtaN + i], tile_w, tile_a[m], vec_scale[i]);
        }
      }
    }
  }

  AccT tile_acc[CtaM * CtaN];
  collapse_tile_acc<Details, CtaM * CtaN>(tile_acc, tile_k_acc);
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
    if (row_zero[m]) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        tile_acc[m * CtaN + i] = static_cast<AccT>(0.f);
      }
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
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA, bool Fast = false>
__global__ void moe_gemv_interleaved_swiglu_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int inter_size, int k,
    cutlass_kernels::ActivationParams activation_params, cutlass_kernels::MoeGemvRowSkipParams row_skip) {
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

  // Zero-weight expert skip: the fc1 output of this row is only ever consumed by the fc2 GEMV for
  // the same permuted row, which skips it too, and finalizeMoeRouting scales it by 0. Write the
  // zeros the SwiGLU epilogue would have written (never leave the slice uninitialized: garbage
  // times a zero scale can be NaN) and bail out before touching any expert weight/scale bytes.
  if (moe_gemv_row_is_zero_weight(row_skip, row)) {
    TypeA* zero_out = out + row * inter_size + static_cast<int>(blockIdx.y) * CtaN * Details::kInterleave / 2;
    for (int ii = static_cast<int>(threadIdx.x); ii < CtaN * Details::kInterleave / 2; ii += Threads) {
      zero_out[ii] = static_cast<TypeA>(0.f);
    }
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
    if constexpr (GroupSize != 0) {
      load_scales<ScalesAccessT, CtaN>(scales_iterator, vec_scale, iter);
    }
    act_iterator.load(tile_a, iter, 0);
    if constexpr (Fast) {
      static_assert(kMoeGemvFp4FastSupported<Details, AccT>);
      alignas(alignof(AccessTypeW)) uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
      float act_f[StepK];
#pragma unroll
      for (int kk = 0; kk < StepK; ++kk) {
        act_f[kk] = static_cast<float>(tile_a[kk]);
      }
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        weight_iterator.load(tile_w_quantized, iter, i);
        fp4_fast_accumulate<Details>(tile_k_acc[i], tile_w_quantized, vec_scale[i], act_f);
      }
    } else {
      AccessTypeW tile_w_quantized[CtaN * Details::kAccessNumW];
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
  }

  AccT tile_acc[CtaM * CtaN];
  collapse_tile_acc<Details, CtaN>(tile_acc, tile_k_acc);
  swiglu_epilogue<Details, CtaM, CtaN, Threads, EnableBias, TypeA, AccT>(out, tile_acc, bias, activation_params);
#endif
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA,
          bool Fast = false>
static void launch_moe_gemv(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                            const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                            int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k,
                            cutlass_kernels::MoeGemvRowSkipParams row_skip, cudaStream_t stream) {
  const int64_t weight_expert_stride = n * k / Details::kElemsPerByteW;
  const int64_t scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT, Fast><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), row_skip);
  } else {
    moe_gemv_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT, Fast><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), row_skip);
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA,
          bool Fast = false>
static void launch_moe_gemv_interleaved_swiglu(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k,
    cutlass_kernels::ActivationParams activation_params, cutlass_kernels::MoeGemvRowSkipParams row_skip,
    cudaStream_t stream) {
  const int64_t n = inter_size * 2;
  const int64_t weight_expert_stride = n * k / Details::kElemsPerByteW;
  const int64_t scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT, Fast>
        <<<grid, block, 0, stream>>>(
            act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
            weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k),
            activation_params, row_skip);
  } else {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT, Fast>
        <<<grid, block, 0, stream>>>(
            act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
            weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k),
            activation_params, row_skip);
  }
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA, bool Fast = false>
static void dispatch_moe_gemv_group_size(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                         const int64_t* expert_first_token_offset,
                                         const int* permuted_row_to_expert, int num_experts,
                                         int64_t expanded_num_rows, int64_t n, int64_t k,
                                         int group_size, cutlass_kernels::MoeGemvRowSkipParams row_skip,
                                         cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv<Details, CtaN, Threads, 0, TypeA, AccT, Fast>(act, weight, scales, bias, out, expert_first_token_offset,
                                                                  permuted_row_to_expert, num_experts, expanded_num_rows, n, k, row_skip, stream);
  } else if (group_size == 16) {
    launch_moe_gemv<Details, CtaN, Threads, 16, TypeA, AccT, Fast>(act, weight, scales, bias, out, expert_first_token_offset,
                                                                   permuted_row_to_expert, num_experts, expanded_num_rows, n, k, row_skip, stream);
  } else if (group_size == 32) {
    launch_moe_gemv<Details, CtaN, Threads, 32, TypeA, AccT, Fast>(act, weight, scales, bias, out, expert_first_token_offset,
                                                                   permuted_row_to_expert, num_experts, expanded_num_rows, n, k, row_skip, stream);
  } else if (group_size == 64) {
    launch_moe_gemv<Details, CtaN, Threads, 64, TypeA, AccT, Fast>(act, weight, scales, bias, out, expert_first_token_offset,
                                                                   permuted_row_to_expert, num_experts, expanded_num_rows, n, k, row_skip, stream);
  } else if (group_size == 128) {
    launch_moe_gemv<Details, CtaN, Threads, 128, TypeA, AccT, Fast>(act, weight, scales, bias, out, expert_first_token_offset,
                                                                    permuted_row_to_expert, num_experts, expanded_num_rows, n, k, row_skip, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA, bool Fast = false>
static void dispatch_moe_gemv_interleaved_swiglu_group_size(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size,
    cutlass_kernels::ActivationParams activation_params, cutlass_kernels::MoeGemvRowSkipParams row_skip,
    cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 0, TypeA, AccT, Fast>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, row_skip, stream);
  } else if (group_size == 16) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 16, TypeA, AccT, Fast>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, row_skip, stream);
  } else if (group_size == 32) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 32, TypeA, AccT, Fast>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, row_skip, stream);
  } else if (group_size == 64) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 64, TypeA, AccT, Fast>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, row_skip, stream);
  } else if (group_size == 128) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 128, TypeA, AccT, Fast>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, row_skip, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
}

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
