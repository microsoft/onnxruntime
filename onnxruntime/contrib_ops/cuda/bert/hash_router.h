// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename I>
Status LaunchHashRouterKernel(
    cudaStream_t stream, T* routing_weights, I* expert_indices, const T* logits,
    const I* input_ids, const I* token_to_expert, int token_count, int num_experts,
    int selected_count, int vocab_size, int score_function, float scaling_factor,
    float epsilon, int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
