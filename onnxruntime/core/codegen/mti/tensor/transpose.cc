// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/transpose.h"

#include <tvm/topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Transpose(const tvm::te::Tensor& X, const tvm::Array<tvm::Integer>& axes, const std::string& name) {
  return tvm::topi::transpose(X, axes, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
