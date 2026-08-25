// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchNgramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
