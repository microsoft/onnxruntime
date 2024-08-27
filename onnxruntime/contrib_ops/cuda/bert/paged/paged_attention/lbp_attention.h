// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/paged/cuda_common.h"

namespace onnxruntime::contrib::paged {

struct DataParallelOutOfPlace {};
struct DataParallelInPlace {};
struct WorkStealing {};

template <typename TIO, typename TKV, typename TSB, typename Sch>
struct LBPAttentionKernel {
  static void launch(
      stream_t stream,
      dev_props_ptr dev_props,
      void* workspace,
      TIO* out_ptr,
      const TIO* q_ptr,
      const TKV* k_cache_ptr,
      const TKV* v_cache_ptr,
      const TSB* scalebias_ptr,
      const int* page_table_ptr,
      const int* context_lens_ptr,
      const float* alibi_slopes_ptr,
      const float scale,
      const int num_seqs,
      const int num_heads,
      const int num_kv_heads,
      const int head_size,
      const int page_size,
      const int max_num_pages_per_seq,
      const int q_stride,
      const int max_context_len
  );

  static void create_workspace(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len);
  static void destroy_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len);
  static void init_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len);
};

}  // namespace onnxruntime::contrib::paged
