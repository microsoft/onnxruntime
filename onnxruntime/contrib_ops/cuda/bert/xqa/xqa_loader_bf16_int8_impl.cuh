// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "xqa_loader.h"
#include <cassert>

// HEAD_ELEMS must be defined by the including file
#ifndef HEAD_ELEMS
#error "HEAD_ELEMS must be defined before including xqa_loader_bf16_int8_impl.cuh"
#endif

// HEAD_DIM_NAMESPACE must be defined by the including file
#ifndef HEAD_DIM_NAMESPACE
#error "HEAD_DIM_NAMESPACE must be defined before including xqa_loader_bf16_int8_impl.cuh"
#endif

// Define global constants for INT8 KV Cache with BF16 Query
#define CACHE_ELEM_ENUM 1
#define USE_PAGED_KV_CACHE 0
#define TOKENS_PER_PAGE 0
#define INPUT_FP16 0  // Q is BF16
#define ALLOW_MULTI_BLOCK_MODE 1

#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 20012

// Include common headers once
#include "cuda_hint.cuh"
#include "mha.h"
// Include all helpers globally to ensure visibility
#include "ldgsts.cuh"
#include "mhaUtils.cuh"
#include "mha_components.cuh"
#include "mma.cuh"
#include "utils.cuh"
#include "hostUtils.h"

// Undefine HEAD_GRP_SIZE and M_TILESIZE to allow re-definition in impl gen
#undef HEAD_GRP_SIZE
#undef M_TILESIZE

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace HEAD_DIM_NAMESPACE {

// ============================================================================
// INT8 KV Cache Instantiations for BF16 Query
// ============================================================================

#define NAMESPACE_NAME grp4_bf16_int8
#define GRP_SIZE 4
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp8_bf16_int8
#define GRP_SIZE 8
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp16_bf16_int8
#define GRP_SIZE 16
#define M_TILESIZE 16
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp32_bf16_int8
#define GRP_SIZE 32
#define M_TILESIZE 32
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

Status LaunchXQAInt8KernelBF16(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    void* workspace,
    size_t workspace_size) {
  int group_size = num_heads / kv_num_heads;
  switch (group_size) {
    case 4:
      return grp4_bf16_int8::Launch<__nv_bfloat16>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
    case 8:
      return grp8_bf16_int8::Launch<__nv_bfloat16>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
    case 16:
      return grp16_bf16_int8::Launch<__nv_bfloat16>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
    case 32:
      return grp32_bf16_int8::Launch<__nv_bfloat16>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA INT8 only supports group_size 4, 8, 16, 32. Input has ", group_size);
  }
}

}  // namespace HEAD_DIM_NAMESPACE
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
