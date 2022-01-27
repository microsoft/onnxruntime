// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/mti_tvm_utils.h"

#include "core/codegen/common/settings.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include <topi/detail/extern.h>
#include <tvm/ir_pass.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::Expr> ToTvmArray(gsl::span<const int64_t> shape) {
  tvm::Array<tvm::Expr> arr;
  for (size_t i = 0; i < shape.size(); ++i) {
    arr.push_back(tvm::Expr(static_cast<int32_t>(shape[i])));
  }
  return arr;
}

tvm::Array<tvm::Integer> ToTvmArrayInt(gsl::span<const int64_t> shape) {
  tvm::Array<tvm::Integer> arr;
  for (size_t i = 0; i < shape.size(); ++i) {
    arr.push_back(shape[i]);
  }
  return arr;
}

tvm::Expr SizeToDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis) {
  tvm::Expr size(1);
  auto rank = shape.size();
  if (static_cast<size_t>(axis) != rank) {
    axis = HandleNegativeAxis(axis, rank);
  }
  for (size_t d = 0; d < std::min(rank, static_cast<size_t>(axis)); ++d)
    size = tvm::ir::Simplify(size * shape[d]);
  return size;
}

tvm::Expr SizeFromDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis) {
  tvm::Expr size(1);
  auto rank = shape.size();
  if (static_cast<size_t>(axis) != rank) {
    axis = HandleNegativeAxis(axis, rank);
  }
  for (size_t d = static_cast<size_t>(axis); d < rank; ++d)
    size = tvm::ir::Simplify(size * shape[d]);
  return size;
}

tvm::Expr RoundUp(tvm::Expr value, tvm::Expr alignment) {
  return tvm::ir::Simplify((value + alignment - 1) / alignment * alignment);
}

tvm::Array<tvm::Expr> ConcatShapes(
    const tvm::Array<tvm::Expr>& shape1,
    const tvm::Array<tvm::Expr>& shape2) {
  tvm::Array<tvm::Expr> result;
  for (size_t i = 0; i < shape1.size(); i++)
    result.push_back(shape1[i]);
  for (size_t i = 0; i < shape2.size(); i++)
    result.push_back(shape2[i]);
  return result;
}

tvm::Tensor Rename(tvm::Tensor X, const std::string& name) {
  const_cast<std::string&>(X->op->name) = name;
  return X;
}

tvm::Array<tvm::Expr> SliceShape(const tvm::Array<tvm::Expr>& shape, const std::vector<int64_t>& axes) {
  tvm::Array<tvm::Expr> new_shape;
  for (auto axis : axes) {
    CHECK(axis < static_cast<int64_t>(shape.size()));
    new_shape.push_back(shape[axis]);
  }
  return new_shape;
}

tvm::Array<tvm::Expr> SliceShapeFromDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis) {
  int64_t rank = static_cast<int64_t>(shape.size());
  axis = HandleNegativeAxis(axis, rank);
  std::vector<int64_t> axes;
  for (auto i = axis; i < rank; ++i)
    axes.push_back(i);
  return SliceShape(shape, axes);
}

tvm::Array<tvm::Expr> SliceShapeToDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis) {
  int64_t rank = static_cast<int64_t>(shape.size());
  axis = HandleNegativeAxis(axis, rank);
  std::vector<int64_t> axes;
  for (auto i = 0; i < axis; ++i)
    axes.push_back(i);
  return SliceShape(shape, axes);
}

bool IsOne(const tvm::Array<tvm::Expr>& shape, int64_t axis) {
  int64_t rank = static_cast<int64_t>(shape.size());
  axis = HandleNegativeAxis(axis, rank);
  const auto& dim = shape[axis];
  auto* p = tvm::as_const_int(dim);
  return p != nullptr && *p == 1;
}

tvm::Tensor Promote(const tvm::Expr& expr, const tvm::Array<tvm::Expr>& shape, const std::string& name) {
  return tvm::compute(
      shape,
      [&](const tvm::Array<tvm::Var>&) {
        return expr;
      },
      name);
}

void DumpTVMModuleToFile(const std::string& filename, tvm::runtime::Module& module) {
  const codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  if (!settings.HasOption(codegen::CodeGenSettings::kCodeGenDumpModule))
    return;

  // ISSUE: note that all option values are converted to lower case. It doesn't cause
  // any issue currently, because all supported formats (i.e. file exts) are of lower case.
  // Just keep in mind that we might have issue if somehow we started to support dump
  // formats with upper case, although it's quite unlikely.
  std::string format = settings.GetOptionValue(codegen::CodeGenSettings::kCodeGenDumpModule);
  std::string module_filename = filename + "." + format;
  module->SaveToFile(module_filename, format);
}

tvm::Tensor MakeZeroTensor(const tvm::Array<tvm::Expr>& shape,
                           HalideIR::Type type,
                           const std::string& name) {
  auto l = [&](const tvm::Array<tvm::Var>& /*indices*/) {
    return tvm::make_zero(type);
  };
  return tvm::compute(shape, l, name);
}

bool BroadcastDim(const tvm::Array<tvm::Expr>& shape, size_t i, size_t output_rank, tvm::Expr& dim) {
  if (i >= output_rank - shape.size()) {
    auto new_dim = shape[shape.size() - output_rank + i];
    if (tvm::ir::Equal(new_dim, dim))
      return true;

    const int64_t* p_new = tvm::as_const_int(new_dim);
    if (p_new != nullptr && *p_new == 1) {
      return true;
    } else {
      const int64_t* p_old = tvm::as_const_int(dim);
      if (p_old != nullptr && *p_old == 1) {
        dim = new_dim;
        return true;
      }
    }
    return false;
  }
  // auto broadcast to outer dims
  return true;
}

tvm::Array<tvm::Tensor> MakeInputsForExtern(const tvm::Array<tvm::Tensor>& inputs, const std::string& name) {
  // note that currently TVM StorageFlatten creates strides like max(symbolic_dim, 1)
  // which is not zero when checking symbolic_dim - max(symbolic_dim, 1)
  // then triggers error like: Trying to bind compact buffer to strided one
  // here's a workaround to reshape inputs to avoid that
  tvm::Array<tvm::Tensor> fixed_inputs;
  for (size_t idx_input = 0; idx_input < inputs.size(); ++idx_input) {
    const auto& input = inputs[idx_input];
    tvm::Array<tvm::Expr> fixed_shape;
    if (input->shape.size() > 0) {
      // stride compute does not use dim 0, so directly push to fixed_shape
      fixed_shape.push_back(input->shape[0]);
      bool need_fix = false;
      for (size_t idx_dim = 1; idx_dim < input->shape.size(); ++idx_dim) {
        const auto& dim = input->shape[idx_dim];
        if (tvm::as_const_int(dim) == nullptr) {
          fixed_shape.push_back(tvm::max(dim, tvm::make_const(HalideIR::Int(32), 1)));
          need_fix = true;
        } else {
          fixed_shape.push_back(dim);
        }
      }
      if (need_fix) {
        fixed_inputs.push_back(tvm_codegen::Reshape(input, fixed_shape, name + "_" + std::to_string(idx_input)));
        continue;
      }
    }
    // no fix needed
    fixed_inputs.push_back(input);
  }
  return fixed_inputs;
}

// Make sure idx is clamped in the range of [-bound, bound - 1]
tvm::Expr ClampIndex(const tvm::Expr& idx, const tvm::Expr& bound) {
  // when idx >= 0, we take tvm::max(..., 0), because (idx < 0) is 0
  // when idx < 0, we take bound + tvm::max(...), because tvm::max(idx, 0) is 0
  return tvm::max(tvm::min(idx, bound - 1), 0) +
         (idx < 0) * (bound + tvm::max(idx, -bound));
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
