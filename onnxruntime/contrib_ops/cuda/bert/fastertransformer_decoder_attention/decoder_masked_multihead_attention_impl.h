// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <
    // The type of the inputs. Supported types: float and half.
    typename T,
    // The type of the QK output. Supported types: float and half.
    typename QK,
    // The hidden dimension per head.
    int head_size,
    // The number of threads per key.
    int THREADS_PER_KEY,
    // The number of threads per value.
    int THREADS_PER_VALUE,
    // The number of threads in a threadblock.
    int THREADS_PER_BLOCK>
__global__ void masked_multihead_attention_kernel(DecoderMaskedMultiHeadAttentionParameters params);

template <typename T, typename QK, int head_size>
void mmha_launch_kernel(const DecoderMaskedMultiHeadAttentionParameters& params, cudaStream_t stream);

inline bool has_decoder_masked_multihead_attention(int sm, int head_size) {
  // This kernel contains some code that cannot be compiled on CUDA ARCH 5.3 or lower
  return (sm >= 53) && (head_size == 32 || head_size == 64 || head_size == 128);
}

}  // namespace cuda

}  // namespace contrib
}  // namespace onnxruntime
