// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchLightningIndexerKernel(
    cudaStream_t stream, int64_t* selected_indices, T* queries, const T* head_weights,
    const T* entries, const int64_t* position_ids, const T* cos_cache, const T* sin_cache,
    int batch_size, int sequence_length, int num_heads, int head_size, int entry_count,
    int index_topk, int compress_rate, int rotary_dim, int cos_cache_width,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
