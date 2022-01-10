// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/crop.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <tvm/topi/nn.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Crop(const tvm::te::Tensor& t,
                 const tvm::Array<tvm::PrimExpr>& border,
                 const tvm::Array<tvm::PrimExpr>& scale,
                 const std::string& name) {
  MTI_ASSERT(t->shape.size() == 4);
   tvm::PrimExpr N = t->shape[0];
   tvm::PrimExpr C = t->shape[1];
   tvm::PrimExpr H = t->shape[2];
   tvm::PrimExpr W = t->shape[3];

  MTI_ASSERT(border.size() == 4);
   tvm::PrimExpr leftBorder = border[0];
   tvm::PrimExpr topBorder = border[1];
   tvm::PrimExpr rightBorder = border[2];
   tvm::PrimExpr bottomBorder = border[3];

   tvm::PrimExpr bottomLimit = H - bottomBorder;
   tvm::PrimExpr rightLimit = W - rightBorder;

  if (!scale.empty()) {
    CHECK_EQ(scale.size(), 2);
    bottomLimit = topBorder + scale[0];
    rightLimit = leftBorder + scale[1];
  }

  tvm::Array<tvm::PrimExpr> output_shape;
  tvm::arith::Analyzer analyzer;
  output_shape.push_back(analyzer.Simplify(N));
  output_shape.push_back(analyzer.Simplify(C));
  output_shape.push_back(analyzer.Simplify(bottomLimit - topBorder));
  output_shape.push_back(analyzer.Simplify(rightLimit - leftBorder));

  auto l = [&](const tvm::Array<tvm::tir::Var>& ovars) {
    tvm::Array<tvm::PrimExpr> indices;

    indices.push_back(tvm::min(ovars[0], output_shape[0] - 1));
    indices.push_back(tvm::min(ovars[1], output_shape[1] - 1));
    indices.push_back(tvm::min(topBorder + ovars[2], topBorder + output_shape[2] - 1));
    indices.push_back(tvm::min(leftBorder + ovars[3], leftBorder + output_shape[3] - 1));

    return t(indices);
  };

  return tvm::te::compute(output_shape, l, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
