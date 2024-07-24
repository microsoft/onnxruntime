// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/tensor.hpp"

#include "contrib_ops/cuda/bert/paged/algorithms.cuh"
#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/type_convert.cuh"
#include "contrib_ops/cuda/bert/paged/warp_utilities.cuh"

namespace onnxruntime::contrib::paged {

using namespace cute;

template <int NumThreads, int HeadSize, int PageSize, int ChunkSize, int VecSize, typename TIO, typename TKV, typename TSB>
__global__ void
__launch_bounds__(NumThreads) reshape_and_cache_kernel(
    TKV* __restrict__ k_cache_out,    // [num_pages, num_heads, head_size/x, page_size, x]
    TKV* __restrict__ v_cache_out,    // [num_pages, num_heads, head_size, page_size]
    TSB* __restrict__ kv_scalebias_out,       // [num_pages, 2, num_heads, 2, num_chunks, page_size], in k_scale, k_bias, v_scale, v_bias order
    const TIO* __restrict__ k_in,             // [num_tokens, num_heads, head_size]
    const TIO* __restrict__ v_in,             // [num_tokens, num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    int num_pages,
    int num_tokens,
    int num_heads,
    int k_in_stride,  // stride of num_tokens dim
    int v_in_stride   // stride of num_tokens dim
)
#if !defined(PAGED_SPLIT_COMPILATION) || defined(RESHAPE_AND_CACHE_KERNEL_IMPL)
{
  using input_t = half;
  using cache_t = float_e4m3_t;
  constexpr int NumInputBits = 8 * sizeof(input_t) * VecSize;
  constexpr int NumOutputBits = 8 * sizeof(cache_t) * VecSize;

  constexpr int NumElemPerCta = NumThreads * VecSize;
  constexpr int HeadSizeNextPowerOfTwo = onnxruntime::contrib::paged::next_power_of_two(HeadSize);
  const int num_heads_per_cta = NumElemPerCta / HeadSizeNextPowerOfTwo;
  const int head_idx_in_cta = threadIdx.x * VecSize / HeadSizeNextPowerOfTwo;
  const int chunk_idx = (threadIdx.x * VecSize % HeadSizeNextPowerOfTwo) / ChunkSize;
  if (head_idx_in_cta >= num_heads) {
    return;
  }

  constexpr int x = 16 / sizeof(cache_t);
  static_assert(x % VecSize == 0);
  static_assert(PageSize % VecSize == 0);
  static_assert(HeadSize % VecSize == 0);

  const bool is_key = blockIdx.y == 0;
  const int64_t token_idx = blockIdx.x / ceil_div(num_heads, num_heads_per_cta);
  const int64_t head_group_idx = blockIdx.x % ceil_div(num_heads, num_heads_per_cta);

  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t page_id = slot_idx / PageSize;
  const int64_t tok_idx_in_page = slot_idx % PageSize;
  const bool is_valid_token = slot_idx >= 0;

  auto gKO = make_tensor(make_gmem_ptr(k_cache_out), make_layout(make_shape(Int<x>{}, Int<PageSize>{}, Int<HeadSize / x>{}, num_heads, num_pages)));
  auto gVO = make_tensor(make_gmem_ptr(v_cache_out), make_layout(make_shape(Int<PageSize>{}, Int<HeadSize>{}, num_heads, num_pages)));
  auto gSB = make_tensor(make_gmem_ptr(kv_scalebias_out), make_layout(make_shape(Int<PageSize>{}, Int<ceil_div(HeadSize, ChunkSize)>{}, _2{}, num_heads, _2{}, num_pages)));
  const auto gKV = make_tensor(
      make_gmem_ptr(is_key ? k_in : v_in),
      make_layout(make_shape(Int<HeadSize>{}, num_heads, num_tokens),make_stride(_1{}, Int<HeadSize>{}, is_key ? k_in_stride : v_in_stride))
  );

#define PAD_HEAD_SIZE_DIM_TO_POWER_OF_2(t) t.compose(make_tile(make_tile(Int<HeadSizeNextPowerOfTwo>{}, _)))

  // ((dim_idx, head_idx_in_cta))
  const auto ctaI = [&]() {
    auto t = group_modes<0, 2>(local_tile(
        gKV,
        make_shape(Int<HeadSize>{}, num_heads_per_cta, _1{}),
        make_coord(_0{}, head_group_idx, token_idx)
    )(_, _, _0{}));
    return PAD_HEAD_SIZE_DIM_TO_POWER_OF_2(t);
  }();
  const auto cta_cI = make_identity_tensor(make_shape(make_shape(Int<HeadSizeNextPowerOfTwo>{}, num_heads_per_cta)));

  // (tok_idx_in_page, chunk_idx, head_idx_in_cta, sb_idx)
  const auto ctaSB = local_tile(
      gSB,
      make_shape(Int<PageSize>{}, shape<1>(gSB), _2{}, num_heads_per_cta, _1{}, _1{}),
      make_coord(_0{}, _0{}, _0{}, head_group_idx, is_key ? 0 : 1, page_id)
  )(_, _, _, _, _0{}, _0{});

  // TODO: we need a tiler to pad dim_idx to next_power_of_two before applying the partition,
  // TODO: conditionally mask out the copy result
  auto tiled_copy = make_tiled_copy(
      Copy_Atom<DefaultCopy, input_t>{},  // placeholder
      make_layout(Int<NumThreads>{}),
      make_layout(Int<VecSize>{})
  );
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  auto tI_view = thr_copy.partition_S(ctaI);  // ((vec), iter), where iter == 1 due to tiled_copy design. TODO: enforce iter == 1
  auto cI = thr_copy.partition_S(cta_cI);
  auto tI = make_tensor<input_t>(Int<VecSize>{});
  auto tI_cvt = make_tensor<float>(Int<VecSize>{});
  auto tO = make_tensor<cache_t>(Int<VecSize>{});

  const bool is_valid_vec = get<0, 0>(cI(_0{})) < HeadSize;

  clear(tI);
  if (is_valid_token && is_valid_vec) {
    copy(AutoVectorizingCopyWithAssumedAlignment<NumInputBits>{}, tI_view, tI);
  }

  CUTE_UNROLL
  for (int i = 0; i < size(tI); i++) {
    tI_cvt(i) = type_convert<float>(tI(i));
  }

  constexpr int GroupSize = ChunkSize / VecSize;

  float acc{};
  int num = is_valid_vec ? VecSize : 0;
  CUTE_UNROLL
  for (int i = 0; i < size(tI); i++) {
    acc += tI_cvt(i);
  }
  acc = warp::reduce<GroupSize>(acc, [](float a, float b) { return a + b; });
  num = warp::reduce<GroupSize>(num, [](int a, int b) { return a + b; });
  // original = a * scaled + b
  float b = acc / num;
  // b = broadcast_in_warp<GroupSize>(b);
  CUTE_UNROLL
  for (int i = 0; i < size(tI); i++) {
    tI_cvt(i) = tI_cvt(i) - b;
  }
  float amax{};
  CUTE_UNROLL
  for (int i = 0; i < size(tI); i++) {
    amax = fmaxf(amax, fabsf(tI_cvt(i)));
  }
  amax = warp::reduce<GroupSize>(amax, [](float a, float b) { return fmaxf(a, fabsf(b)); });
  float target_max = 416.0f;
  // float a = amax / target_max;
  // // a = broadcast_in_warp<GroupSize>(a);
  // float scale = __frcp_rn(a);
  float scale = target_max / amax;
  float a = __frcp_rn(scale);
  CUTE_UNROLL
  for (int i = 0; i < size(tI); i++) {
    tO(i) = fast_type_convert<float_e4m3_t>(__float2half(tI_cvt(i) * scale));
  }

  // additional computing is not allowed after return!
  if (!is_valid_token || !is_valid_vec) {
    return;
  }
  if (is_key) {
    // (((dim_idx), head_idx_in_cta))
    auto ctaO = [&]() {
      auto t = local_tile(  // [x, head_size/x, num_heads_per_cta]
          gKO,
          make_shape(shape<0>(gKO), _1{}, shape<2>(gKO), num_heads_per_cta, _1{}),
          make_coord(_0{}, tok_idx_in_page, _0{}, head_group_idx, page_id)
      )(_, _0{}, _, _, _0{});
      auto t2 = group_modes<0, 2>(group_modes<0, 2>(t));  // [((x, head_size/x), num_heads_per_cta)]
      return PAD_HEAD_SIZE_DIM_TO_POWER_OF_2(t2);
    }();

    auto tO_view = thr_copy.partition_D(ctaO);
    copy(AutoVectorizingCopyWithAssumedAlignment<NumOutputBits>{}, tO, tO_view);
  } else {
    // ((dim_idx, head_idx_in_cta))
    auto ctaO = [&]() {
      auto t = group_modes<0, 2>(local_tile(  // [head_size, num_heads_per_cta]
          gVO,
          make_shape(_1{}, shape<1>(gVO), num_heads_per_cta, _1{}),
          make_coord(tok_idx_in_page, _0{}, head_group_idx, page_id)
      )(_0{}, _, _, _0{}));
      return PAD_HEAD_SIZE_DIM_TO_POWER_OF_2(t);
    }();

    auto tO_view = thr_copy.partition_D(ctaO);
    copy(AutoVectorizingCopyWithAssumedAlignment<NumOutputBits>{}, tO, tO_view);
  }
  if (threadIdx.x % GroupSize == 0) {
    auto tSB = ctaSB(tok_idx_in_page, chunk_idx, _, head_idx_in_cta);
    tSB(_0{}) = type_convert<half>(a);
    tSB(_1{}) = type_convert<half>(b);
  }
#undef PAD_HEAD_SIZE_DIM_TO_POWER_OF_2
}
#else
    ;
#endif

}  // namespace onnxruntime::contrib::paged
