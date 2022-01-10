// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/mti_tvm_utils.h"

#include "core/codegen/common/settings.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include <tvm/topi/detail/extern.h>
#include <tvm/topi/broadcast.h>
#include <tvm/arith/analyzer.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::PrimExpr> ToTvmArray(const std::vector<int64_t>& shape) {
  tvm::Array<tvm::PrimExpr> arr;
  for (size_t i = 0; i < shape.size(); ++i) {
    arr.push_back( tvm::PrimExpr(static_cast<int32_t>(shape[i])));
  }
  return arr;
}

tvm::Array<tvm::Integer> ToTvmArrayInt(const std::vector<int64_t>& shape) {
  tvm::Array<tvm::Integer> arr;
  for (size_t i = 0; i < shape.size(); ++i) {
    arr.push_back(shape[i]);
  }
  return arr;
}

 tvm::PrimExpr SizeToDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis) {
   tvm::PrimExpr size(1);
  auto rank = shape.size();
  tvm::arith::Analyzer analyzer;
  if (static_cast<size_t>(axis) != rank) {
    axis = HandleNegativeAxis(axis, rank);
  }
  for (size_t d = 0; d < std::min(rank, static_cast<size_t>(axis)); ++d)
    size = analyzer.Simplify(size * shape[d]);
  return size;
}

 tvm::PrimExpr SizeFromDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis) {
   tvm::PrimExpr size(1);
  auto rank = shape.size();
  tvm::arith::Analyzer analyzer;
  if (static_cast<size_t>(axis) != rank) {
    axis = HandleNegativeAxis(axis, rank);
  }
  for (size_t d = static_cast<size_t>(axis); d < rank; ++d)
    size = analyzer.Simplify(size * shape[d]);
  return size;
}

 tvm::PrimExpr RoundUp( tvm::PrimExpr value,  tvm::PrimExpr alignment) {
  tvm::arith::Analyzer analyzer;
  return analyzer.Simplify((value + alignment - 1) / alignment * alignment);
}

tvm::Array<tvm::PrimExpr> ConcatShapes(
    const tvm::Array<tvm::PrimExpr>& shape1,
    const tvm::Array<tvm::PrimExpr>& shape2) {
  tvm::Array<tvm::PrimExpr> result;
  for (size_t i = 0; i < shape1.size(); i++)
    result.push_back(shape1[i]);
  for (size_t i = 0; i < shape2.size(); i++)
    result.push_back(shape2[i]);
  return result;
}

tvm::te::Tensor Rename(tvm::te::Tensor X, const std::string& name) {
  const_cast<std::string&>(X->op->name) = name;
  return X;
}

tvm::Array<tvm::PrimExpr> SliceShape(const tvm::Array<tvm::PrimExpr>& shape, const std::vector<int64_t>& axes) {
  tvm::Array<tvm::PrimExpr> new_shape;
  for (auto axis : axes) {
    CHECK(axis < static_cast<int64_t>(shape.size()));
    new_shape.push_back(shape[axis]);
  }
  return new_shape;
}

tvm::Array<tvm::PrimExpr> SliceShapeFromDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis) {
  int64_t rank = static_cast<int64_t>(shape.size());
  axis = HandleNegativeAxis(axis, rank);
  std::vector<int64_t> axes;
  for (auto i = axis; i < rank; ++i)
    axes.push_back(i);
  return SliceShape(shape, axes);
}

tvm::Array<tvm::PrimExpr> SliceShapeToDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis) {
  int64_t rank = static_cast<int64_t>(shape.size());
  axis = HandleNegativeAxis(axis, rank);
  std::vector<int64_t> axes;
  for (auto i = 0; i < axis; ++i)
    axes.push_back(i);
  return SliceShape(shape, axes);
}

bool IsOne(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis) {
  int64_t rank = static_cast<int64_t>(shape.size());
  axis = HandleNegativeAxis(axis, rank);
  const auto& dim = shape[axis];
  auto* p = tvm::tir::as_const_int(dim);
  return p != nullptr && *p == 1;
}

tvm::te::Tensor Promote(const  tvm::PrimExpr& expr, const tvm::Array<tvm::PrimExpr>& shape, const std::string& name) {
  return tvm::te::compute(
      shape,
      [&](const tvm::Array<tvm::tir::Var>&) {
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

tvm::te::Tensor MakeZeroTensor(const tvm::Array<tvm::PrimExpr>& shape,
                           tvm::DataType type,
                           const std::string& name) {
  auto l = [&](const tvm::Array<tvm::tir::Var>& /*indices*/) {
    return tvm::tir::make_zero(type);
  };
  return tvm::te::compute(shape, l, name);
}

bool BroadcastDim(const tvm::Array<tvm::PrimExpr>& shape, size_t i, size_t output_rank, tvm::PrimExpr& dim) {
  if (i >= output_rank - shape.size()) {
    auto new_dim = shape[shape.size() - output_rank + i];
    // if (tvm::equal(new_dim, dim))
    //   return true;

    const int64_t* p_new = tvm::tir::as_const_int(new_dim);
    if (p_new != nullptr && *p_new == 1) {
      return true;
    } else {
      const int64_t* p_old = tvm::tir::as_const_int(dim);
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

tvm::Array<tvm::te::Tensor> MakeInputsForExtern(const tvm::Array<tvm::te::Tensor>& inputs, const std::string& name) {
  // note that currently TVM StorageFlatten creates strides like max(symbolic_dim, 1)
  // which is not zero when checking symbolic_dim - max(symbolic_dim, 1)
  // then triggers error like: Trying to bind compact buffer to strided one
  // here's a workaround to reshape inputs to avoid that
  tvm::Array<tvm::te::Tensor> fixed_inputs;
  for (size_t idx_input = 0; idx_input < inputs.size(); ++idx_input) {
    const auto& input = inputs[idx_input];
    tvm::Array<tvm::PrimExpr> fixed_shape;
    if (input->shape.size() > 0) {
      // stride compute does not use dim 0, so directly push to fixed_shape
      fixed_shape.push_back(input->shape[0]);
      bool need_fix = false;
      for (size_t idx_dim = 1; idx_dim < input->shape.size(); ++idx_dim) {
        const auto& dim = input->shape[idx_dim];
        if (tvm::tir::as_const_int(dim) == nullptr) {
          fixed_shape.push_back(tvm::max(dim, tvm::tir::make_const(tvm::DataType::Int(32), 1)));
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
 tvm::PrimExpr ClampIndex(const  tvm::PrimExpr& idx, const  tvm::PrimExpr& bound) {
  // when idx >= 0, we take tvm::max(..., 0), because (idx < 0) is 0
  // when idx < 0, we take bound + tvm::max(...), because tvm::max(idx, 0) is 0
  return tvm::max(tvm::min(idx, bound - 1), 0) +
         (idx < 0) * (bound + tvm::max(idx, -bound));
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
