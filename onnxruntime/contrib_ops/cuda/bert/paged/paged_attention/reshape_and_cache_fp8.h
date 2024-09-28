// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cute/config.hpp"
#include "cute/numeric/numeric_types.hpp"

#include "contrib_ops/cuda/bert/paged/cuda_common.h"

namespace onnxruntime::contrib::paged {

void launch_reshape_and_cache_fp8(
    stream_t stream,
    dev_props_ptr dev_props,
    cute::float_e4m3_t* k_cache_out,    // [num_pages,    num_heads, head_size/x, page_size, x]
    cute::float_e4m3_t* v_cache_out,    // [num_pages,    num_heads, head_size,   page_size]
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
);

}  // namespace onnxruntime::contrib::paged
