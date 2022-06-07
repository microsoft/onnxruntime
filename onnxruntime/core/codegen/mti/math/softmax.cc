// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/softmax.h"

#include "core/codegen/mti/tensor/reshape_ops.h"
#include <topi/nn/softmax.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Softmax(const tvm::Tensor& input, int64_t axis, const std::string& name) {
  tvm::Tensor flatten_t = Flatten(input, axis, "softmax_flatten");
  return Reshape(topi::nn::softmax(flatten_t, 1, name), input->shape, "softmax_reshape");
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
