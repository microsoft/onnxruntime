// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

void launchNormalizeKernel( const float* input,
   float* output,
   float* gamma_ptr,  // gamma
   float* beta_ptr,   // beta
   int nBatch,
   int sequence_len,
   int encode_len );

}  // namespace cuda
}  //namespace contrib
}  // namespace onnxruntime
