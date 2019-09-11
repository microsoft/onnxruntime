// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
  size_t getAttentionWorkspaceSize(size_t wordSize, int batchsize, int numHeads, int headSize, int sequenceLength);

  void launchAttentionKernel(
   const float* input, // input tensor
   const int* mask,    // Mask per word
   float* output,      // output tensor
   int batchSize,      // batch size (B)
   int sequenceLength, // sequence length (S)
   int numHeads,       // number of attention heads (N)
   int headSize,       // hidden layer size per head (H)
   void* workspace,    // work space
   cublasHandle_t& cublas);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
