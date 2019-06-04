// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/math/softmax.h"

#include "core/providers/nuphar/mti_x86/math/softmax_internal.h"

namespace onnxruntime {
namespace nuphar_codegen {

tvm::Tensor Softmax(const tvm::Tensor& input, int64_t axis, const std::string& name) {
  return internal::SoftmaxInternal(input, axis, name, /*logarithmic*/ false);
}

}  // namespace nuphar_codegen
}  // namespace onnxruntime
