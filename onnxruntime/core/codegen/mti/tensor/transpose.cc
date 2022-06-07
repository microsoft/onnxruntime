// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/transpose.h"

#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Transpose(const tvm::Tensor& X, const tvm::Array<tvm::Integer>& axes, const std::string& name) {
  return topi::transpose(X, axes, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
