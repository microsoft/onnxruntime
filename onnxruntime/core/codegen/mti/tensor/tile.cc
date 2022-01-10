// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/tile.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Tile(const tvm::te::Tensor& t,
                 const std::vector<int64_t>& repeats,
                 const std::string& name) {
  MTI_ASSERT(repeats.size() == t->shape.size());
  tvm::Array<tvm::PrimExpr> output_shape;

  bool repeats_zero = false;
  for (size_t i = 0; i < t->shape.size(); ++i) {
    if (repeats[i] == 0)
      repeats_zero = true;
    output_shape.push_back(t->shape[i] * gsl::narrow<int>(repeats[i]));
  }

  auto l = [&](const tvm::Array<tvm::tir::Var>& ovars) {
    if (repeats_zero)
      return tvm::tir::make_zero(t->dtype);

    tvm::Array<tvm::PrimExpr> ivars;
    for (size_t i = 0; i < t->shape.size(); ++i) {
       tvm::PrimExpr ovar = ovars[i];
      ivars.push_back(tvm::floormod(ovar, t->shape[i]));
    }
    return t(ivars);
  };

  return tvm::te::compute(output_shape, l, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
