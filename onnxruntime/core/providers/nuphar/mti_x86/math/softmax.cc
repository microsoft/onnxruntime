// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/math/softmax.h"

#include "core/providers/nuphar/mti_x86/math/softmax_internal.h"

namespace onnxruntime {
namespace nuphar {

tvm::Tensor Softmax(const tvm::Tensor& input, int64_t axis, int64_t vector_width, const std::string& name) {
  return internal::SoftmaxInternal(input, axis, vector_width, name, /*logarithmic*/ false);
}

}  // namespace nuphar
}  // namespace onnxruntime
