// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/split.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/common/gsl.h"
#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

// Similar to numpy, topi::split takes split indices rather than the
// sizes of the splits. Thus we implement our own.
tvm::Array<tvm::Tensor> Split(const tvm::Tensor& X,
                              const tvm::Array<tvm::Expr>& split_sizes,
                              int64_t axis,
                              const std::string& name) {
  MTI_ASSERT(axis < gsl::narrow<int64_t>(X->shape.size()));
  size_t axis_t = gsl::narrow<int>(axis);

  tvm::Array<tvm::Array<tvm::Expr>> output_shapes;
  int num_splits = gsl::narrow<int>(split_sizes.size());
  for (auto& s : split_sizes) {
    tvm::Array<tvm::Expr> shape;
    for (size_t i = 0; i < axis_t; i++) {
      shape.push_back(X->shape[i]);
    }
    shape.push_back(s);
    for (size_t i = axis_t + 1; i < X->shape.size(); i++) {
      shape.push_back(X->shape[i]);
    }
    output_shapes.push_back(shape);
  }

  tvm::Array<tvm::Tensor> res;
  int idx = 0;
  for (int i_split = 0; i_split < num_splits; ++i_split) {
    tvm::Expr s = split_sizes[i_split];
    auto l = [&](const tvm::Array<tvm::Var>& indices) {
      tvm::Array<tvm::Expr> new_indices;
      for (size_t i = 0; i < axis_t; i++) {
        new_indices.push_back(indices[i]);
      }
      new_indices.push_back(indices[axis_t] + idx);
      for (size_t i = axis_t + 1; i < X->shape.size(); i++) {
        new_indices.push_back(indices[i]);
      }
      MTI_ASSERT(topi::detail::IsConstInt(s));
      MTI_ASSERT(new_indices.size() == X->shape.size());
      int size = topi::detail::GetConstInt(s);
      idx += size;
      return X(new_indices);
    };
    res.push_back(tvm::compute(output_shapes[i_split], l, name));
  }

  MTI_ASSERT(topi::detail::IsConstInt(X->shape[axis_t]));
  int size_of_splitted_axis = static_cast<int>(topi::detail::GetConstInt(X->shape[axis_t]));
  MTI_ASSERT(idx == size_of_splitted_axis);
  return res;
}

tvm::Array<tvm::Tensor> SplitWithIndices(const tvm::Tensor& X,
                                         const tvm::Array<tvm::Integer>& split_sizes,
                                         int64_t axis,
                                         const std::string& name) {
  return topi::split(X, split_sizes, gsl::narrow<int>(axis), name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
