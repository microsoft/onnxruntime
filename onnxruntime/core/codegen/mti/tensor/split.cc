// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/split.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "gsl/gsl"
#include <tvm/topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

// Similar to numpy, tvm::topi::split takes split indices rather than the
// sizes of the splits. Thus we implement our own.
tvm::Array<tvm::te::Tensor> Split(const tvm::te::Tensor& X,
                              const tvm::Array<tvm::PrimExpr>& split_sizes,
                              int64_t axis,
                              const std::string& name) {
  MTI_ASSERT(axis < gsl::narrow<int64_t>(X->shape.size()));
  size_t axis_t = gsl::narrow<int>(axis);

  tvm::Array<tvm::Array<tvm::PrimExpr>> output_shapes;
  int num_splits = gsl::narrow<int>(split_sizes.size());
  for (auto& s : split_sizes) {
    tvm::Array<tvm::PrimExpr> shape;
    for (size_t i = 0; i < axis_t; i++) {
      shape.push_back(X->shape[i]);
    }
    shape.push_back(s);
    for (size_t i = axis_t + 1; i < X->shape.size(); i++) {
      shape.push_back(X->shape[i]);
    }
    output_shapes.push_back(shape);
  }

  tvm::Array<tvm::te::Tensor> res;
  int idx = 0;
  for (int i_split = 0; i_split < num_splits; ++i_split) {
     tvm::PrimExpr s = split_sizes[i_split];
    auto l = [&](const tvm::Array<tvm::tir::Var>& indices) {
      tvm::Array<tvm::PrimExpr> new_indices;
      for (size_t i = 0; i < axis_t; i++) {
        new_indices.push_back(indices[i]);
      }
      new_indices.push_back(indices[axis_t] + idx);
      for (size_t i = axis_t + 1; i < X->shape.size(); i++) {
        new_indices.push_back(indices[i]);
      }
      MTI_ASSERT(tvm::topi::detail::IsConstInt(s));
      MTI_ASSERT(new_indices.size() == X->shape.size());
      int size = tvm::topi::detail::GetConstInt(s);
      idx += size;
      return X(new_indices);
    };
    res.push_back(tvm::te::compute(output_shapes[i_split], l, name));
  }

  MTI_ASSERT(tvm::topi::detail::IsConstInt(X->shape[axis_t]));
  int size_of_splitted_axis = static_cast<int>(tvm::topi::detail::GetConstInt(X->shape[axis_t]));
  MTI_ASSERT(idx == size_of_splitted_axis);
  return res;
}

tvm::Array<tvm::te::Tensor> SplitWithIndices(const tvm::te::Tensor& X,
                                             const tvm::Array<tvm::Integer>& split_sizes,
                                             int64_t axis,
                                             const std::string& name) {
  tvm::Array<tvm::PrimExpr> split_size_expr;
  for(auto it: split_sizes) {
    split_size_expr.push_back(tvm::tir::make_const(tvm::DataType::Int(64), (int64_t)it));
  }
  return tvm::topi::split(X, split_size_expr, gsl::narrow<int>(axis), name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
