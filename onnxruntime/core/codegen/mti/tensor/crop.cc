// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/crop.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/nn.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Crop(const tvm::Tensor& t,
                 const tvm::Array<tvm::Expr>& border,
                 const tvm::Array<tvm::Expr>& scale,
                 const std::string& name) {
  MTI_ASSERT(t->shape.size() == 4);
  tvm::Expr N = t->shape[0];
  tvm::Expr C = t->shape[1];
  tvm::Expr H = t->shape[2];
  tvm::Expr W = t->shape[3];

  MTI_ASSERT(border.size() == 4);
  tvm::Expr leftBorder = border[0];
  tvm::Expr topBorder = border[1];
  tvm::Expr rightBorder = border[2];
  tvm::Expr bottomBorder = border[3];

  tvm::Expr bottomLimit = H - bottomBorder;
  tvm::Expr rightLimit = W - rightBorder;

  if (!scale.empty()) {
    CHECK_EQ(scale.size(), 2);
    bottomLimit = topBorder + scale[0];
    rightLimit = leftBorder + scale[1];
  }

  tvm::Array<tvm::Expr> output_shape;
  output_shape.push_back(tvm::ir::Simplify(N));
  output_shape.push_back(tvm::ir::Simplify(C));
  output_shape.push_back(tvm::ir::Simplify(bottomLimit - topBorder));
  output_shape.push_back(tvm::ir::Simplify(rightLimit - leftBorder));

  auto l = [&](const tvm::Array<tvm::Var>& ovars) {
    tvm::Array<tvm::Expr> indices;

    indices.push_back(tvm::min(ovars[0], output_shape[0] - 1));
    indices.push_back(tvm::min(ovars[1], output_shape[1] - 1));
    indices.push_back(tvm::min(topBorder + ovars[2], topBorder + output_shape[2] - 1));
    indices.push_back(tvm::min(leftBorder + ovars[3], leftBorder + output_shape[3] - 1));

    return t(indices);
  };

  return tvm::compute(output_shape, l, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
