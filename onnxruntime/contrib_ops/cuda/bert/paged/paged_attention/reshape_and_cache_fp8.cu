// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cute/tensor.hpp"

#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/paged_attention/reshape_and_cache.cuh"

namespace onnxruntime::contrib::paged {

using namespace cute;

void launch_reshape_and_cache_fp8(
    stream_t stream,
    dev_props_ptr dev_props,
    float_e4m3_t* k_cache_out,    // [num_pages,    num_heads, head_size/x, page_size, x]
    float_e4m3_t* v_cache_out,    // [num_pages,    num_heads, head_size,   page_size]
    half* kv_scalebias_out,       // [num_pages, 2, num_heads, 2, num_chunks,  page_size]
    const half* k_in,             // [num_tokens,   num_heads, head_size]
    const half* v_in,             // [num_tokens,   num_heads, head_size]
    const int64_t* slot_mapping,  // [num_tokens]
    int num_pages,
    int num_tokens,
    int num_heads,
    int head_size,
    int page_size,
    int k_in_stride,
    int v_in_stride
) {
  constexpr int ChunkSize = 32;
  constexpr int VecSize = 8 / sizeof(half);  // target LDG.64
  constexpr int NumThreads = 128;
  constexpr int NumElemPerCta = NumThreads * VecSize;

#define LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(HEAD_SIZE, PAGE_SIZE)                                    \
  const int num_heads_per_cta = NumElemPerCta / onnxruntime::contrib::paged::next_power_of_two(head_size);                 \
  const int num_ctas = num_tokens * ceil_div(num_heads, num_heads_per_cta);                                \
  reshape_and_cache_kernel<NumThreads, HEAD_SIZE, PAGE_SIZE, ChunkSize, VecSize, half, float_e4m3_t, half> \
      <<<dim3(num_ctas, /*0 for k, 1 for v*/ 2), NumThreads, 0, stream>>>(                                 \
          k_cache_out, v_cache_out, kv_scalebias_out,                                                      \
          k_in, v_in, slot_mapping,                                                                        \
          num_pages, num_tokens, num_heads,                                                                \
          k_in_stride, v_in_stride                                                                         \
      );                                                                                                   \
  break;

  do {
    if (page_size == 32) {
      if (head_size == 64) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(64, 32);
      } else if (head_size == 80) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(80, 32);
      } else if (head_size == 96) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(96, 32);
      } else if (head_size == 112) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(112, 32);
      } else if (head_size == 128) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(128, 32);
      } else if (head_size == 256) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(256, 32);
      } else {
        throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
      }
    } else if (page_size == 16) {
      if (head_size == 64) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(64, 16);
      } else if (head_size == 80) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(80, 16);
      } else if (head_size == 96) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(96, 16);
      } else if (head_size == 112) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(112, 16);
      } else if (head_size == 128) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(128, 16);
      } else if (head_size == 256) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(256, 16);
      } else {
        throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
      }
    } else if (page_size == 8) {
      if (head_size == 64) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(64, 8);
      } else if (head_size == 80) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(80, 8);
      } else if (head_size == 96) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(96, 8);
      } else if (head_size == 112) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(112, 8);
      } else if (head_size == 128) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(128, 8);
      } else if (head_size == 256) {
        LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK(256, 8);
      } else {
        throw std::runtime_error(std::string("Unsupported head size: ") + std::to_string(head_size));
      }
    } else {
      throw std::runtime_error(std::string("Unsupported page size: ") + std::to_string(page_size));
    }
  } while (0);
#undef LAUNCH_RESHAPE_AND_CACHE_KERNEL_AND_BREAK
}

}  // namespace onnxruntime::contrib::paged
