// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

Status LaunchEmbedLayerNormKernel(cudaStream_t stream,
                                  void* output,                        // output tensor
                                  void* mask_index,                    // output mask index
                                  const int* input_ids,                // input word IDs
                                  const int* segment_ids,              // input segment IDs
                                  const int* input_mask,               // input mask
                                  const void* gamma,                   // weight for layer normalization
                                  const void* beta,                    // bias for layer normalization
                                  const void* word_embedding,          // weights for word embeddings
                                  const void* position_embedding,      // weights for position embeddings
                                  const void* segment_embedding,       // weights for segment (like sentence) embeddings
                                  float epsilon,                       // epsilon for layer normalization
                                  const int hidden_size,               // hidden size (that is head_size * num_heads)
                                  int batch_size,                      // batch size
                                  int sequence_length,                 // sequence length
                                  const size_t element_size,           // size of output element: 2 for half, 4 for float.
                                  void* embedding_sum,                 // Optional output of sum of embeddings
                                  const int* position_ids,             // Optional input of position ids
                                  const bool broadcast_position_ids);  // Whether to broadcast position ids

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
