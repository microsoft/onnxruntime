// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/weight_layout/weight_layout.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/utils.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"

namespace onnxruntime {
namespace tvm_codegen {

static tvm::Tensor CreateTVMPlaceholder(
    const std::string& name,
    HalideIR::Type type,
    int dim) {
  tvm::Array<tvm::Expr> shape;
  if (dim > 0) {
    for (int i = 0; i < dim; ++i) {
      shape.push_back(tvm::Var(name + "_v" + std::to_string(i)));
    }
  } else {
    shape.push_back(1);
  }
  return tvm::placeholder(shape, type, name + "_placeholder");
}

const std::string WeightLayout::GetKey(
    const std::string& name,
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    int input_dim,
    float pad_zero) {
  std::ostringstream key;
  key << name << "_type_" << static_cast<int>(proto_type);
  key << "_dim_" << input_dim;
  key << "_pad_zero_" << pad_zero;
  return NormalizeCppName(key.str());
}

WeightLayout::WeightLayout(
    const std::string& name,
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    int input_dim,
    float pad_zero)
    : name_(GetKey(name, proto_type, input_dim, pad_zero)),
      proto_type_(proto_type),
      input_dim_(input_dim),
      pad_zero_(pad_zero) {}

const std::string& WeightLayout::Name() const {
  return name_;
}

void WeightLayout::CreateLayoutMarshallingTVMOp(tvm::Array<tvm::Tensor>& inputs,
                                                tvm::Array<tvm::Tensor>& outputs) const {
  HalideIR::Type halide_type = ToTvmType(proto_type_);

  tvm::Tensor placeholder = CreateTVMPlaceholder(name_, halide_type, input_dim_);
  inputs.push_back(placeholder);

  tvm::Array<tvm::Expr> new_shape = ToActualShape(placeholder);
  CoordTransFunc new_coord_to_old_coord_func = ToNominal(placeholder);
  tvm::Expr pad_zero_expr = tvm::make_const(halide_type, pad_zero_);

  tvm::Tensor output = tvm::compute(
      new_shape,
      [&](const tvm::Array<tvm::Var>& output_coord) {
        tvm::Array<tvm::Expr> output_coord1;
        for (const auto& coord : output_coord)
          output_coord1.push_back(coord);
        auto input_coord = new_coord_to_old_coord_func(output_coord1);
        ORT_ENFORCE(input_coord.size() == placeholder->shape.size());

        if (input_coord.size() > 0) {
          auto in_range = (input_coord[0] >= 0) && (input_coord[0] < placeholder->shape[0]);
          for (size_t dim = 1; dim < input_coord.size(); ++dim)
            in_range = in_range && (input_coord[dim] >= 0) && (input_coord[dim] < placeholder->shape[dim]);

          return tvm::if_then_else(in_range, placeholder(input_coord), pad_zero_expr);
        } else {
          // scalar
          return placeholder(input_coord);
        }
      });

  outputs.push_back(output);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
