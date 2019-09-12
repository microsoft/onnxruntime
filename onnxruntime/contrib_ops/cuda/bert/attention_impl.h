// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
  size_t getAttentionWorkspaceSize(size_t element_size, int batchsize, int num_heads, int head_size, int sequence_length);

  void launchAttentionKernel(
   const void* input,         // input tensor
   const int* mask,           // Mask per word
   void* output,              // output tensor
   int batch_size,            // batch size (B)
   int sequence_length,       // sequence length (S)
   int num_heads,             // number of attention heads (N)
   int head_size,             // hidden layer size per head (H)
   void* workspace,           // work space
   cublasHandle_t& cublas,    // cublas handle
   const size_t element_size  // element size of input tensor
   );

  }  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
