// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Shared body of the paged-KV XQA translation units. Each xqa_paged_<query>_<cache>_<head>.cu
// defines the macros below and includes this file; the macros pick the KV element type, the query
// element type, the head size and the exported entry-point name.
//
// Expected macros:
//   HEAD_ELEMS            head size (64 / 128 / 256)
//   HEAD_DIM_NAMESPACE    namespace for the head size (H64 / H128 / H256)
//   XQA_PAGED_CACHE_ELEM  0 = FP16/BF16 KV cache, 1 = INT8 KV cache, 2 = FP8 KV cache
//   XQA_PAGED_INPUT_FP16  1 = FP16 query/output, 0 = BF16
//   XQA_PAGED_QUERY_T     query element type (half / __nv_bfloat16)
//   XQA_PAGED_FAMILY      token used to make the per-TU namespaces unique (e.g. fp16_int8)
//   XQA_PAGED_LAUNCH_FN   name of the exported dispatcher for this (query, cache) pair

#pragma once
#include "xqa_paged_loader.h"
#include <cassert>

#ifndef HEAD_ELEMS
#error "HEAD_ELEMS must be defined before including xqa_paged_loader_impl.cuh"
#endif
#ifndef HEAD_DIM_NAMESPACE
#error "HEAD_DIM_NAMESPACE must be defined before including xqa_paged_loader_impl.cuh"
#endif
#ifndef XQA_PAGED_CACHE_ELEM
#error "XQA_PAGED_CACHE_ELEM must be defined before including xqa_paged_loader_impl.cuh"
#endif
#ifndef XQA_PAGED_INPUT_FP16
#error "XQA_PAGED_INPUT_FP16 must be defined before including xqa_paged_loader_impl.cuh"
#endif
#ifndef XQA_PAGED_QUERY_T
#error "XQA_PAGED_QUERY_T must be defined before including xqa_paged_loader_impl.cuh"
#endif
#ifndef XQA_PAGED_FAMILY
#error "XQA_PAGED_FAMILY must be defined before including xqa_paged_loader_impl.cuh"
#endif
#ifndef XQA_PAGED_LAUNCH_FN
#error "XQA_PAGED_LAUNCH_FN must be defined before including xqa_paged_loader_impl.cuh"
#endif

#define CACHE_ELEM_ENUM XQA_PAGED_CACHE_ELEM
#define INPUT_FP16 XQA_PAGED_INPUT_FP16
// Paged KV cache, vLLM/SGLang layout: separate K and V pools, each page laid out as
// [tokens_per_page, kv_num_heads, head_size]. This matches PagedAttention's block pool exactly.
#define TOKENS_PER_PAGE 128
#define USE_PAGED_KV_CACHE 1
#define PAGED_KV_CACHE_LAYOUT 1
#define ALLOW_MULTI_BLOCK_MODE 1
// Compiled with sliding-window support so one kernel serves both global attention
// (local_window_size == -1, mapped to a window >= max_seq_len) and sliding-window models.
#define SLIDING_WINDOW 1

#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 20012

#include "cuda_hint.cuh"
#include "mha.h"
#include "ldgsts.cuh"
#include "mhaUtils.cuh"
#include "mha_components.cuh"
#include "mma.cuh"
#include "utils.cuh"
#include "hostUtils.h"

#undef HEAD_GRP_SIZE
#undef M_TILESIZE

// Token-pasting helpers so a single TU body can produce unique namespace names per
// (query dtype, cache dtype) family.
#define XQA_PAGED_CAT_(a, b, c) a##b##c
#define XQA_PAGED_CAT(a, b, c) XQA_PAGED_CAT_(a, b, c)
#define XQA_PAGED_NS(grp) XQA_PAGED_CAT(grp, XQA_PAGED_FAMILY, _paged)

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace HEAD_DIM_NAMESPACE {

#if !defined(XQA_PAGED_GROUP6_ONLY)
#define NAMESPACE_NAME XQA_PAGED_NS(grp4_)
#define GRP_SIZE 4
#define M_TILESIZE 8
#include "xqa_paged_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE
#endif

#define NAMESPACE_NAME XQA_PAGED_NS(grp6_)
#define GRP_SIZE 6
#define M_TILESIZE 8
#include "xqa_paged_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#if !defined(XQA_PAGED_GROUP6_ONLY)
#define NAMESPACE_NAME XQA_PAGED_NS(grp8_)
#define GRP_SIZE 8
#define M_TILESIZE 8
#include "xqa_paged_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME XQA_PAGED_NS(grp16_)
#define GRP_SIZE 16
#define M_TILESIZE 16
#include "xqa_paged_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME XQA_PAGED_NS(grp32_)
#define GRP_SIZE 32
#define M_TILESIZE 32
#include "xqa_paged_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE
#endif

#define XQA_PAGED_DISPATCH(grp)                                                                      \
  return XQA_PAGED_NS(grp)::Launch<XQA_PAGED_QUERY_T>(                                               \
      device_prop, stream, query, key_cache, value_cache, output, page_table, batch_size, num_heads, \
      kv_num_heads, head_size, max_pages_per_seq, scale, local_window_size, past_seq_lens,           \
      attention_sinks, k_cache_scale, v_cache_scale, workspace, workspace_size)

Status XQA_PAGED_LAUNCH_FN(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int* page_table,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_pages_per_seq,
    const float scale,
    const int local_window_size,
    const int* past_seq_lens,
    const float* attention_sinks,
    const float* k_cache_scale,
    const float* v_cache_scale,
    void* workspace,
    size_t workspace_size) {
  const int group_size = num_heads / kv_num_heads;
  switch (group_size) {
#if !defined(XQA_PAGED_GROUP6_ONLY)
    case 4:
      XQA_PAGED_DISPATCH(grp4_);
#endif
    case 6:
      XQA_PAGED_DISPATCH(grp6_);
#if !defined(XQA_PAGED_GROUP6_ONLY)
    case 8:
      XQA_PAGED_DISPATCH(grp8_);
    case 16:
      XQA_PAGED_DISPATCH(grp16_);
    case 32:
      XQA_PAGED_DISPATCH(grp32_);
#endif
    default:
#if defined(XQA_PAGED_GROUP6_ONLY)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "This paged XQA specialization only supports group_size 6. Input has ", group_size);
#else
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Paged XQA only supports group_size 4, 6, 8, 16, 32. Input has ", group_size);
#endif
  }
}

// Shared-memory requirement of the instantiation that would actually run, so the caller can fall
// back when it exceeds the device's per-block opt-in limit. See xqa_impl_gen.cuh::GetSmemSize.
size_t XQA_PAGED_CAT(XQA_PAGED_LAUNCH_FN, _, SmemSize)(const int num_heads, const int kv_num_heads) {
  const int group_size = num_heads / kv_num_heads;
  switch (group_size) {
#if !defined(XQA_PAGED_GROUP6_ONLY)
    case 4:
      return XQA_PAGED_NS(grp4_)::GetSmemSize();
#endif
    case 6:
      return XQA_PAGED_NS(grp6_)::GetSmemSize();
#if !defined(XQA_PAGED_GROUP6_ONLY)
    case 8:
      return XQA_PAGED_NS(grp8_)::GetSmemSize();
    case 16:
      return XQA_PAGED_NS(grp16_)::GetSmemSize();
    case 32:
      return XQA_PAGED_NS(grp32_)::GetSmemSize();
#endif
    default:
      return 0;
  }
}

#undef XQA_PAGED_DISPATCH

}  // namespace HEAD_DIM_NAMESPACE
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
