// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// is_valid_scratch must point to at least one int32_t of device memory. The launcher uses it to
// record whether cumulative_sequence_length is globally well-formed (cu_seqlens[0] == 0,
// cu_seqlens[batch_size] == total_tokens, and strictly increasing in between) before any output-
// producing kernel runs, so that a malformed array can never cause two requests to race on the
// same output element: every output element is written by exactly one of the validation-gated
// kernels, never both.
template <typename T>
Status LaunchVarlenNGramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const int32_t* cu_seqlens,
    const T* past_ids,
    const T* head_offsets,
    const T* eos_token_id,
    const int32_t* segment_ids,
    T* output,
    T* present_ids,
    int64_t batch_size,
    int64_t total_tokens,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    bool reset_on_eos,
    int max_threads_per_block,
    int32_t* is_valid_scratch);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
