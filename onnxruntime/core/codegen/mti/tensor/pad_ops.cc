// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/pad_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/nn.h>

namespace onnxruntime {
namespace tvm_codegen {

// Note topi::pad does not support modes {edge, reflect}
// Therefore, MTI implements a generic Pad
tvm::Tensor Pad(const tvm::Tensor& t,
                const tvm::Array<tvm::Expr>& pad_before,
                const tvm::Array<tvm::Expr>& pad_after,
                float pad_value,
                const std::string& mode,
                const std::string& name) {
  MTI_ASSERT(pad_before.size() >= 1);
  MTI_ASSERT(pad_before.size() == pad_after.size());
  MTI_ASSERT(pad_before.size() == t->shape.size());

  tvm::Array<tvm::Expr> output_shape;
  for (size_t i = 0; i < t->shape.size(); ++i) {
    output_shape.push_back(
        tvm::ir::Simplify(t->shape[i] + pad_before[i] + pad_after[i]));
  }

  auto l = [&](const tvm::Array<tvm::Var>& ovars) {
    tvm::Array<tvm::Expr> conds;
    tvm::Array<tvm::Expr> indices;
    tvm::Array<tvm::Expr> coords;

    for (size_t i = 0; i < t->shape.size(); ++i) {
      tvm::Expr ivar = ovars[i] - pad_before[i];
      tvm::Expr min = 0;
      tvm::Expr extent = t->shape[i];

      conds.push_back(ivar < min);
      conds.push_back(ivar >= min + extent);
      indices.push_back(tvm::max(tvm::min(ivar, min + extent - 1), min));

      if (mode == "reflect") {
        // calculate indices for reflect mode
        tvm::Expr limit = extent - 1;
        tvm::Expr coord = ivar - min;
        // Avoid mod zero when tensor shape has 1,
        // e.g. input shape is [1, 3, 3] instead of [3, 3]
        auto* p_limit = tvm::as_const_int(limit);
        if (p_limit != nullptr && *p_limit != 0)
          coord = (coord + 2 * limit) % (2 * limit);  // avoid negative value
        coord = coord - limit;
        coord = tvm::abs(coord);
        coord = limit - coord;
        coord = coord + min;
        coords.push_back(coord);
      }
    }

    if (mode == "reflect") {
      return tvm::ir::Select::make(topi::detail::Map(conds, tvm::ir::Or::make),
                                   t(coords), t(indices));
    } else if (mode == "constant") {
      return tvm::ir::Select::make(topi::detail::Map(conds, tvm::ir::Or::make),
                                   tvm::make_const(t->dtype, pad_value), t(indices));
    }

    // default mode is edge
    return t(indices);
  };

  return tvm::compute(output_shape, l, name);
}

tvm::Tensor Pad(const tvm::Tensor& t,
                const tvm::Array<tvm::Expr>& output_shape,
                const tvm::Expr& pad_value,
                const std::string& name) {
  MTI_ASSERT(t->dtype == pad_value.type());

  auto l = [&](const tvm::Array<tvm::Var>& ovars) {
    tvm::Array<tvm::Expr> conds;
    tvm::Array<tvm::Expr> indices;

    for (size_t i = 0; i < t->shape.size(); ++i) {
      tvm::Expr ivar = ovars[i];
      tvm::Expr min = 0;
      tvm::Expr extent = t->shape[i];

      conds.push_back(ivar < min);
      conds.push_back(ivar >= min + extent);
      indices.push_back(tvm::max(tvm::min(ivar, min + extent - 1), min));
    }

    return tvm::ir::Select::make(topi::detail::Map(conds, tvm::ir::Or::make),
                                 pad_value, t(indices));
  };

  return tvm::compute(output_shape, l, name);
}

tvm::Tensor PadLastDim(const tvm::Tensor& t,
                       const int32_t align_size,
                       const tvm::Expr& pad_value,
                       const std::string& name) {
  auto input_shape = t->shape;
  tvm::Array<tvm::Expr> out_shape;
  size_t input_shape_rank = input_shape.size();
  for (size_t i = 0; i < input_shape_rank - 1; ++i) {
    out_shape.push_back(input_shape[i]);
  }
  out_shape.push_back(
      (input_shape[input_shape_rank - 1] + align_size - 1) /
      align_size * align_size);

  return Pad(t, out_shape, pad_value, name + "_pad");
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
