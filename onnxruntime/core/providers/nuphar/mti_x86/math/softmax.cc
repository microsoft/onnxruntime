// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/math/softmax.h"

#include "core/providers/nuphar/mti_x86/math/softmax_internal.h"

namespace onnxruntime {
namespace nuphar {

tvm::Tensor Softmax(const tvm::Tensor& input, int64_t axis, tvm_codegen::CodeGenContext& ctx_codegen, const std::string& name) {
  return internal::SoftmaxInternal(input, axis, ctx_codegen, name, /*logarithmic*/ false);
}

}  // namespace nuphar
}  // namespace onnxruntime
