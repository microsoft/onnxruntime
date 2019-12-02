// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/shape_op.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Shape(const tvm::Tensor& X, const std::string& name) {
  int ndim = static_cast<int>(X->shape.size());
  tvm::Array<tvm::Expr> out_shape{ndim};
  return tvm::compute(
      out_shape, [&](const tvm::Array<tvm::Var>& indices) {
        auto idx = indices[0];
        tvm::Expr ret = 0;
        for (int i = 0; i < ndim; ++i) {
          ret = tvm::ir::Select::make(idx == i, X->shape[i], ret);
        }
        return tvm::cast(HalideIR::Int(64), ret);
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
