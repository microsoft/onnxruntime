// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/tile.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/common/gsl.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Tile(const tvm::Tensor& t,
                 const std::vector<int64_t>& repeats,
                 const std::string& name) {
  MTI_ASSERT(repeats.size() == t->shape.size());
  tvm::Array<tvm::Expr> output_shape;

  bool repeats_zero = false;
  for (size_t i = 0; i < t->shape.size(); ++i) {
    if (repeats[i] == 0)
      repeats_zero = true;
    output_shape.push_back(t->shape[i] * gsl::narrow<int>(repeats[i]));
  }

  auto l = [&](const tvm::Array<tvm::Var>& ovars) {
    if (repeats_zero)
      return tvm::make_zero(t->dtype);

    tvm::Array<tvm::Expr> ivars;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      tvm::Expr ovar = ovars[i];
      ivars.push_back(ovar % t->shape[i]);
    }
    return t(ivars);
  };

  return tvm::compute(output_shape, l, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
