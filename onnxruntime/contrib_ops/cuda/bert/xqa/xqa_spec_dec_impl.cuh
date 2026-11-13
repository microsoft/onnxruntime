// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Shared body of the contiguous-cache speculative-decoding XQA translation units. Each
// xqa_spec_dec_<query>_<head>.cu defines the macros below and includes this file.
//
// Expected macros:
//   HEAD_ELEMS          head size
//   HEAD_DIM_NAMESPACE  namespace for the head size (e.g. H128)
//   XQA_SPEC_INPUT_FP16 1 = FP16 query/output/cache, 0 = BF16
//   XQA_SPEC_QUERY_T    query element type (half / __nv_bfloat16)
//   XQA_SPEC_FAMILY     token that makes the per-TU namespaces unique (e.g. bf16)
//   XQA_SPEC_LAUNCH_FN  name of the exported dispatcher

#pragma once

#define SPEC_DEC 1
// Linear (chain) drafts: a uniform causal mask over the new tokens rather than a draft tree.
#define IS_SPEC_DEC_TREE 0
#define XQA_SPEC_DEC_SPLIT_PER_TOKEN_BLOCK 1

#include "xqa_loader.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cassert>

#define USE_PAGED_KV_CACHE 0
#define TOKENS_PER_PAGE 0
#define CACHE_ELEM_ENUM 0
#define INPUT_FP16 XQA_SPEC_INPUT_FP16
#define ALLOW_MULTI_BLOCK_MODE 1
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

#define XQA_SPEC_CAT_(a, b, c) a##b##c
#define XQA_SPEC_CAT(a, b, c) XQA_SPEC_CAT_(a, b, c)
#define XQA_SPEC_NS(grp) XQA_SPEC_CAT(grp, XQA_SPEC_FAMILY, _spec_dec)

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace HEAD_DIM_NAMESPACE {

// One CTA covers M_TILESIZE (query head, token) rows of a KV head group, so the KV tile it loads is shared by
// M_TILESIZE / group_size query tokens instead of being reloaded per query head.
#define NAMESPACE_NAME XQA_SPEC_NS(grp4_)
#define GRP_SIZE 4
#define M_TILESIZE 32
#include "xqa_spec_dec_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME XQA_SPEC_NS(grp8_)
#define GRP_SIZE 8
#define M_TILESIZE 32
#include "xqa_spec_dec_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME XQA_SPEC_NS(grp16_)
#define GRP_SIZE 16
#define M_TILESIZE 32
#include "xqa_spec_dec_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

Status XQA_SPEC_LAUNCH_FN(const cudaDeviceProp& device_prop, cudaStream_t stream, const void* query,
                          const void* key_cache, const void* value_cache, void* output, const int batch_size,
                          const int sequence_length, const int num_heads, const int kv_num_heads,
                          const int max_seq_len, const float scale, const int local_window_size, const bool is_bsnh,
                          const int* past_seq_lens, void* workspace, size_t workspace_size) {
  switch (num_heads / kv_num_heads) {
#define XQA_SPEC_LAUNCH                                                                                   \
  Launch<XQA_SPEC_QUERY_T>(device_prop, stream, query, key_cache, value_cache, output, batch_size,        \
                           sequence_length, kv_num_heads, max_seq_len, scale, local_window_size, is_bsnh, \
                           past_seq_lens, workspace, workspace_size)
    case 4:
      return XQA_SPEC_NS(grp4_)::XQA_SPEC_LAUNCH;
    case 8:
      return XQA_SPEC_NS(grp8_)::XQA_SPEC_LAUNCH;
    case 16:
      return XQA_SPEC_NS(grp16_)::XQA_SPEC_LAUNCH;
#undef XQA_SPEC_LAUNCH
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Speculative XQA supports group_size 4, 8, 16. Input has ",
                             num_heads / kv_num_heads);
  }
}

size_t XQA_SPEC_CAT(XQA_SPEC_LAUNCH_FN, _, WorkspaceSize)(const cudaDeviceProp& device_prop, const int batch_size,
                                                          const int sequence_length, const int num_heads,
                                                          const int kv_num_heads, const int max_seq_len) {
  switch (num_heads / kv_num_heads) {
    case 4:
      return XQA_SPEC_NS(grp4_)::GetWorkspaceSize(device_prop, batch_size, kv_num_heads, sequence_length, max_seq_len);
    case 8:
      return XQA_SPEC_NS(grp8_)::GetWorkspaceSize(device_prop, batch_size, kv_num_heads, sequence_length, max_seq_len);
    case 16:
      return XQA_SPEC_NS(grp16_)::GetWorkspaceSize(device_prop, batch_size, kv_num_heads, sequence_length,
                                                   max_seq_len);
    default:
      return 0;
  }
}

size_t XQA_SPEC_CAT(XQA_SPEC_LAUNCH_FN, _, SmemSize)(const int num_heads, const int kv_num_heads) {
  switch (num_heads / kv_num_heads) {
    case 4:
      return XQA_SPEC_NS(grp4_)::GetSmemSize();
    case 8:
      return XQA_SPEC_NS(grp8_)::GetSmemSize();
    case 16:
      return XQA_SPEC_NS(grp16_)::GetSmemSize();
    default:
      return 0;
  }
}

}  // namespace HEAD_DIM_NAMESPACE
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
