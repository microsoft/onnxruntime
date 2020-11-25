// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/reshape_ops.h"

#include "core/codegen/mti/common.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Flatten(const tvm::Tensor& X, int64_t axis, const std::string& name) {
  const auto& input_shape = X->shape;
  return Reshape(X, {SizeToDimension(input_shape, axis), SizeFromDimension(input_shape, axis)}, name);
}

tvm::Tensor Identity(const tvm::Tensor& X, const std::string& name) {
  return Reshape(X, X->shape, name);
}

tvm::Tensor Reshape(const tvm::Tensor& X, const tvm::Array<tvm::Expr>& new_shape, const std::string& name) {
  if (new_shape.size() > 0) {
    auto X_dim = SizeToDimension(X->shape, X->shape.size());
    auto new_dim = SizeToDimension(new_shape, new_shape.size());
    auto* pX_dim = tvm::as_const_int(X_dim);
    auto* pNew_dim = tvm::as_const_int(new_dim);

    if (pX_dim != nullptr && pNew_dim != nullptr) {
      MTI_ASSERT(*pX_dim == *pNew_dim);
    }
    return topi::reshape(X, new_shape, name);
  } else {
    // generate empty dim tensor with origial input data value
    tvm::Array<tvm::Expr> tmp_shape;
    tmp_shape.push_back(1);
    auto tmp_tensor = topi::reshape(X, tmp_shape);
    return tvm::compute(
        new_shape,
        [&](const tvm::Array<tvm::Var>&) {
          return tmp_tensor[0];
        },
        name);
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
