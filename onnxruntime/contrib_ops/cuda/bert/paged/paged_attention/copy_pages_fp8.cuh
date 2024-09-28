// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/config.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/int_tuple.hpp"

#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"

namespace onnxruntime::contrib::paged {

using namespace cute;

__global__ void copy_pages_fp8_kernel(
    int64_t* k_cache_ptrs,
    int64_t* v_cache_ptrs,
    int64_t* kv_scalebias_ptrs,
    const int64_t* __restrict__ page_mapping,
    const int num_k_elems_per_page,
    const int num_v_elems_per_page,
    const int num_sb_elems_per_page
) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  float_e4m3_t* k_cache = reinterpret_cast<float_e4m3_t*>(k_cache_ptrs[layer_idx]);
  float_e4m3_t* v_cache = reinterpret_cast<float_e4m3_t*>(v_cache_ptrs[layer_idx]);
  half* kv_scalebias = reinterpret_cast<half*>(kv_scalebias_ptrs[layer_idx]);

  const int64_t src_page_id = page_mapping[2 * pair_idx];
  const int64_t dst_page_id = page_mapping[2 * pair_idx + 1];

  for (int i = threadIdx.x; i < num_k_elems_per_page; i += blockDim.x) {
    int64_t src_offset = src_page_id * num_k_elems_per_page + i;
    int64_t dst_offset = dst_page_id * num_k_elems_per_page + i;
    k_cache[dst_offset] = k_cache[src_offset];
  }
  for (int i = threadIdx.x; i < num_v_elems_per_page; i += blockDim.x) {
    int64_t src_offset = src_page_id * num_v_elems_per_page + i;
    int64_t dst_offset = dst_page_id * num_v_elems_per_page + i;
    v_cache[dst_offset] = v_cache[src_offset];
  }

  for (int i = threadIdx.x; i < num_sb_elems_per_page; i += blockDim.x) {
    int64_t src_offset = src_page_id * num_sb_elems_per_page + i;
    int64_t dst_offset = dst_page_id * num_sb_elems_per_page + i;
    kv_scalebias[dst_offset] = kv_scalebias[src_offset];
  }
}

void launch_copy_pages_fp8_kernel(
    stream_t stream,
    int64_t* k_cache_ptrs,                     // [num_layers] of ptrs, each ptr points to num_k_elems_per_page  of fp8
    int64_t* v_cache_ptrs,                     // [num_layers] of ptrs, each ptr points to num_v_elems_per_page  of fp8
    int64_t* kv_scalebias_ptrs,                // [num_layers] of ptrs, each ptr points to num_sb_elems_per_page of half (k_scale, k_bias, v_scale, v_bias)
    const int64_t* __restrict__ page_mapping,  // [num_pairs, 2], aka, num_pairs of (src, dst)
    const int num_layers,
    const int num_pairs,
    const int num_heads,
    const int head_size,
    const int page_size
) {
  constexpr int x = 16 / sizeof(float_e4m3_t);
  constexpr int ChunkSize = 32;
  const int num_k_elems_per_page = num_heads * (head_size / x) * page_size * x;
  const int num_v_elems_per_page = num_heads * head_size * page_size;
  const int num_sb_elems_per_page = 4 * num_heads * ceil_div(head_size, ChunkSize) * page_size;

  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, num_v_elems_per_page));
  copy_pages_fp8_kernel<<<grid, block, 0, stream>>>(
      k_cache_ptrs,
      v_cache_ptrs,
      kv_scalebias_ptrs,
      page_mapping,
      num_k_elems_per_page,
      num_v_elems_per_page,
      num_sb_elems_per_page
  );
}

}  // namespace onnxruntime::contrib::paged
