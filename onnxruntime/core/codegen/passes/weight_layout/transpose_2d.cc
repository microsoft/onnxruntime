// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/weight_layout/transpose_2d.h"

#include "core/codegen/passes/utils/codegen_context.h"

namespace onnxruntime {
namespace tvm_codegen {

constexpr auto local_layout_name = "transpose_2d";

const std::string WeightLayoutTranspose2D::GetKey(
    ONNX_NAMESPACE::TensorProto_DataType proto_type) {
  return WeightLayout::GetKey(local_layout_name, proto_type, 2, 0.0f);
}

WeightLayoutTranspose2D::WeightLayoutTranspose2D(
    ONNX_NAMESPACE::TensorProto_DataType proto_type)
    : WeightLayout(local_layout_name, proto_type, 2, 0.0f) {}

CoordTransFunc WeightLayoutTranspose2D::ToActual(const tvm::Tensor& /*X*/) const {
  return [&](const tvm::Array<tvm::Expr>& nominal_coord) {
    ORT_ENFORCE(nominal_coord.size() == 2);
    const auto& y = nominal_coord[0];
    const auto& x = nominal_coord[1];
    return tvm::Array<tvm::Expr>{
        x,
        y};
  };
}

CoordTransFunc WeightLayoutTranspose2D::ToNominal(const tvm::Tensor& /*X*/) const {
  return [&](const tvm::Array<tvm::Expr>& actual_coord) {
    ORT_ENFORCE(actual_coord.size() == 2);
    const auto& y = actual_coord[0];
    const auto& x = actual_coord[1];
    return tvm::Array<tvm::Expr>{
        x,
        y};
  };
}

tvm::Array<tvm::Expr> WeightLayoutTranspose2D::ToActualShape(const tvm::Tensor& X) const {
  tvm::Array<tvm::Expr> new_shape = {
      X->shape[1],
      X->shape[0]};
  return new_shape;
}

std::vector<int64_t> WeightLayoutTranspose2D::ToActualShape(const Tensor* X) const {
  ORT_ENFORCE(X != nullptr);
  ORT_ENFORCE(X->Shape().GetDims().size() == 2);
  auto old_shape = X->Shape().GetDims();

  std::vector<int64_t> new_shape = {
      old_shape[1],
      old_shape[0]};

  return new_shape;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
