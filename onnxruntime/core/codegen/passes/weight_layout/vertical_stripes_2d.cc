// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/weight_layout/vertical_stripes_2d.h"

#include "core/codegen/passes/utils/codegen_context.h"

namespace onnxruntime {
namespace tvm_codegen {

constexpr auto local_name_prefix = "vertical_stripe_2d_";

const std::string WeightLayoutVerticalStripe2D::GetKey(
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    int stripe_width) {
  return WeightLayout::GetKey(
      local_name_prefix + std::to_string(stripe_width),
      proto_type, 2, 0.0f);
}

WeightLayoutVerticalStripe2D::WeightLayoutVerticalStripe2D(
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    int stripe_width)
    : WeightLayout(
          local_name_prefix + std::to_string(stripe_width),
          proto_type, 2, 0.0f),
      stripe_width_(stripe_width) {
}

CoordTransFunc WeightLayoutVerticalStripe2D::ToActual(const tvm::Tensor& /*X*/) const {
  return [&](const tvm::Array<tvm::Expr>& nominal_coord) {
    ORT_ENFORCE(nominal_coord.size() == 2);
    const auto& y = nominal_coord[0];
    const auto& x = nominal_coord[1];
    return tvm::Array<tvm::Expr>{
        x / stripe_width_,
        y,
        x % stripe_width_};
  };
}

CoordTransFunc WeightLayoutVerticalStripe2D::ToNominal(const tvm::Tensor& /*X*/) const {
  return [&](const tvm::Array<tvm::Expr>& actual_coord) {
    ORT_ENFORCE(actual_coord.size() == 3);
    const auto& z = actual_coord[0];
    const auto& y = actual_coord[1];
    const auto& x = actual_coord[2];
    return tvm::Array<tvm::Expr>{
        y,
        x + stripe_width_ * z};
  };
}

tvm::Array<tvm::Expr> WeightLayoutVerticalStripe2D::ToActualShape(const tvm::Tensor& X) const {
  tvm::Array<tvm::Expr> new_shape = {
      (X->shape[1] + stripe_width_ - 1) / stripe_width_,
      X->shape[0],
      stripe_width_};
  return new_shape;
}

std::vector<int64_t> WeightLayoutVerticalStripe2D::ToActualShape(const Tensor* X) const {
  ORT_ENFORCE(X != nullptr);
  auto old_shape = X->Shape().GetDims();

  ORT_ENFORCE(old_shape.size() == 2);

  std::vector<int64_t> new_shape = {
      (old_shape[1] + stripe_width_ - 1) / stripe_width_,
      old_shape[0],
      stripe_width_};

  return new_shape;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
