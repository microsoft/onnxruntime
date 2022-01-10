// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/logsoftmax.h"

#include "core/codegen/mti/tensor/reshape_ops.h"
#include <tvm/topi/nn/softmax.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor LogSoftmax(const tvm::te::Tensor& input, int64_t axis, const std::string& name) {
  tvm::te::Tensor flatten_t = Flatten(input, axis, "logsoftmax_flatten");
  return Reshape(tvm::topi::nn::log_softmax(flatten_t, name), input->shape, "logsoftmax_reshape");
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
