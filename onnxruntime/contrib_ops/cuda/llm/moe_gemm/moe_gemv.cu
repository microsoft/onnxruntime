// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// MoE batched GEMV kernel. One thread-block (CtaM = 1 row) processes a single
// expanded row; the row's expert determines the weight/scale/bias base pointers.
// Mirrors the dense fpA_intB_gemv `kernel<>` body for GroupSize=0 (per-channel),
// EnableActScale=false, EnableZero=false, ApplyAlphaInAdvance=false.
template <typename Details, int CtaN, int Threads, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void moe_gemv_kernel(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                int64_t const* expert_first_token_offset, int num_experts,
                                int64_t weight_expert_stride, int n, int k) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);

  int const row = blockIdx.x;

  // Resolve the expert owning this permuted row via a small linear scan over the
  // monotonically increasing offsets (num_experts is small, value broadcasts).
  int expert = 0;
#pragma unroll 1
  for (int e = 0; e < num_experts; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
    } else {
      break;
    }
  }

  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * n;
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
  // Per-channel scales: constant along K (step 0), indexed by output column.
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(scales, real_offset_n, 0, Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  TypeA tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

  TypeA vec_scale[CtaN];
#pragma unroll
  for (int i = 0; i < CtaN; ++i) {
    scales_iterator.load(vec_scale + i, 0, i);
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      act_iterator.load(tile_a, iter, i);
      mma<Details, 1, CtaN, StepK>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
    }
  }
  epilogue<Details, CtaM, CtaN, Threads, EnableBias, false>(out, n, tile_acc, bias, 1.0f);
#endif
}

template <typename Details, int CtaN, int Threads, typename TypeA>
static void launch_moe_gemv(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                            int64_t const* expert_first_token_offset, int num_experts, int64_t expanded_num_rows,
                            int64_t n, int64_t k, cudaStream_t stream) {
  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_kernel<Details, CtaN, Threads, true><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, num_experts, weight_expert_stride,
        static_cast<int>(n), static_cast<int>(k));
  } else {
    moe_gemv_kernel<Details, CtaN, Threads, false><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, num_experts, weight_expert_stride,
        static_cast<int>(n), static_cast<int>(k));
  }
}

}  // namespace fpA_intB_gemv

namespace moe_gemv {

namespace fiv = onnxruntime::llm::kernels::fpA_intB_gemv;

// CtaN/Threads match the dense per-channel (EnableZero=false) m-tile config.
static constexpr int kCtaN = 8;
static constexpr int kThreads = 128;
// int4 ColumnMajorInterleave (Sm80) tile width along N.
static constexpr int kTileSizeK = 64;
static constexpr int kInt4Interleave = 128 * 8 / (kTileSizeK * 4);  // = 4

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k) {
  if (sm < 80) {
    return false;
  }
  if (expanded_num_rows <= 0 || expanded_num_rows > 64) {
    return false;
  }
  // n must tile evenly; k must tile evenly into StepK (32 for int4) along interleaved-K.
  if (n % (kCtaN * kInt4Interleave) != 0) {
    return false;
  }
  int64_t const interleaved_k = k * kInt4Interleave;
  int const step_k = 128 / 4;  // kStepK for int4
  if (interleaved_k % step_k != 0) {
    return false;
  }
  return true;
}

template <typename T>
struct DetailsForT;

template <>
struct DetailsForT<half> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
};

template <>
struct DetailsForT<__nv_bfloat16> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
};

template <typename T>
void launch_moe_gemv_int4_per_channel(T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
                                      int64_t const* expert_first_token_offset, int num_experts,
                                      int64_t expanded_num_rows, int64_t n, int64_t k, int sm, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForT<T>::Details;
  using TypeA = typename DetailsForT<T>::TypeA;
  fiv::launch_moe_gemv<Details, kCtaN, kThreads, TypeA>(
      const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
      const_cast<uint8_t*>(weight),
      const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
      const_cast<TypeA*>(reinterpret_cast<TypeA const*>(bias)),
      reinterpret_cast<TypeA*>(out),
      expert_first_token_offset, num_experts, expanded_num_rows, n, k, stream);
}

template void launch_moe_gemv_int4_per_channel<half>(half const*, uint8_t const*, half const*, half const*, half*,
                                                     int64_t const*, int, int64_t, int64_t, int64_t, int, cudaStream_t);
template void launch_moe_gemv_int4_per_channel<__nv_bfloat16>(__nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*,
                                                              __nv_bfloat16 const*, __nv_bfloat16*, int64_t const*, int,
                                                              int64_t, int64_t, int64_t, int, cudaStream_t);

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
