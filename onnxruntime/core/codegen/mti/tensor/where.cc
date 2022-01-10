// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/where.h"

#include <tvm/topi/broadcast.h>
#include <tvm/topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Where(const tvm::te::Tensor& B,
                  const tvm::te::Tensor& X,
                  const tvm::te::Tensor& Y,
                  const std::string& name) {
  size_t rank = std::max(std::max(B->shape.size(), X->shape.size()), Y->shape.size());
  tvm::Array<tvm::PrimExpr> output_shape;
  for (size_t i = 0; i < rank; ++i) {
     tvm::PrimExpr dim = tvm::tir::make_const(tvm::DataType::Int(32), 1);
    bool broadcasted =
        BroadcastDim(B->shape, i, rank, dim) &&
        BroadcastDim(X->shape, i, rank, dim) &&
        BroadcastDim(Y->shape, i, rank, dim);
    MTI_ASSERT(broadcasted);
    output_shape.push_back(dim);
  }

  return tvm::topi::where(tvm::topi::broadcast_to(B, output_shape),
                          tvm::topi::broadcast_to(X, output_shape),
                          tvm::topi::broadcast_to(Y, output_shape),
                          name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
