// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define QORDERED_UNARY_OP_SHARED_MEMORY_DECLARATION(name) \
  Status QOrderedUnarySharedMemory_##name(                \
      cudaStream_t stream,                                \
      const int8_t* input_data,                           \
      const float* input_scale,                           \
      int8_t* output_data,                                \
      const float* output_scale,                          \
      size_t count)

QORDERED_UNARY_OP_SHARED_MEMORY_DECLARATION(Gelu);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
