
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define LIST_OF_QORDER_UNARY_OPS()          \
  QORDER_UNARY_OP_NAME_EXPR(Gelu, _Gelu(a)) \
  QORDER_UNARY_OP_NAME_EXPR(Erf, _Erf(a))

#define QORDER_UNARY_OP_DECLARATION(name) \
  void QOrderUnary_##name(                \
      cudaStream_t stream,                \
      const int8_t* input_data,           \
      const float* input_scale,           \
      int8_t* output_data,                \
      const float* output_scale,          \
      size_t count)

#define QORDER_UNARY_OP_NAME_EXPR(name, expr) QORDER_UNARY_OP_DECLARATION(name);
LIST_OF_QORDER_UNARY_OPS()
#undef QORDER_UNARY_OP_NAME_EXPR

}  // namespace cuda
}  // namespace conrtib
}  // namespace onnxruntime
