
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <inttypes.h>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define QORDER_UNARY_OP_DECLARATION(name) \
  void QOrderUnary_##name(                \
      cudaStream_t stream,                \
      const int8_t* input_data,           \
      const float* input_scale,           \
      int8_t* output_data,                \
      const float* output_scale,          \
      size_t count)

#define QORDER_UNARY_OP_SHAREMEMORY_DECLARATION(name) \
  void QOrderUnaryShareMemory_##name(                 \
      cudaStream_t stream,                            \
      const int8_t* input_data,                       \
      const float* input_scale,                       \
      int8_t* output_data,                            \
      const float* output_scale,                      \
      size_t count)

QORDER_UNARY_OP_DECLARATION(Gelu);

QORDER_UNARY_OP_SHAREMEMORY_DECLARATION(Gelu);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
