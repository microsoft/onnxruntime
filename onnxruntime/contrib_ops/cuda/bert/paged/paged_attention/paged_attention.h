// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/paged/cuda_common.h"

namespace onnxruntime::contrib::paged {

template <typename TIO, typename TKV, typename TSB = void>
void launch_paged_attention_kernel(
    stream_t stream,
    dev_props_ptr dev_props,
    TIO* out,
    const TIO* q,
    const TKV* k_cache,
    const TKV* v_cache,
    const TSB* scalebias,
    const int* page_table,
    const int* context_lens,
    const float* alibi_slopes,
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

}  // namespace onnxruntime::contrib::paged
