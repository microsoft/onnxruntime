// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <
    typename T,
    typename QuantParamT,
    int BLOCK_SIZE,
    int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(
    const cudaStream_t stream,
    T* out,
    const T* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const int* head_mapping,
    float scale,
    const int* block_tables,
    const int max_num_blocks_per_seq,
    const int* context_lens,
    int max_context_len,
    const float* alibi_slopes_ptr,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    int kv_quant_chunk_size,
    const QuantParamT* kv_quant_params_cache);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
