// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchVarlenNGramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const int32_t* cu_seqlens,
    const T* past_ids,
    T* output,
    T* present_ids,
    int64_t batch_size,
    int64_t total_tokens,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
