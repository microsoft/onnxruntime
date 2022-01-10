// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/shape_op.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Shape(const tvm::te::Tensor& X, const std::string& name) {
  int ndim = static_cast<int>(X->shape.size());
  tvm::Array<tvm::PrimExpr> out_shape{ndim};
  return tvm::te::compute(
      out_shape, [&](const tvm::Array<tvm::tir::Var>& indices) {
        auto idx = indices[0];
         tvm::PrimExpr ret = 0;
        for (int i = 0; i < ndim; ++i) {
          ret = tvm::tir::Select(idx == i, X->shape[i], ret);
        }
        return tvm::cast(tvm::DataType::Int(64), ret);
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
