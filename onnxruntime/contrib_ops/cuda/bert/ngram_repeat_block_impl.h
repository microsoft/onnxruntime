// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

void NGramRepeatBlockImpl(
    cudaStream_t stream,
    const long* tokens_ptr,
    const float* scores_ptr,
    int bsz,
    int step,
    int max_predict_len,
    int vocab_size,
    int beam_size,
    int no_repeat_ngram_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
