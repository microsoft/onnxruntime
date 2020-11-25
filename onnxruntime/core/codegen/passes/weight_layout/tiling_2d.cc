// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/weight_layout/tiling_2d.h"

#include "core/codegen/passes/utils/codegen_context.h"

namespace onnxruntime {
namespace tvm_codegen {

constexpr auto local_name_prefix = "tiling_2d_";
constexpr int num_bits = 8;

const std::string WeightLayoutTiling2D::GetKey(
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    int vector_width) {
  return WeightLayout::GetKey(
      local_name_prefix + std::to_string(vector_width),
      proto_type, 2, 0.0f);
}

WeightLayoutTiling2D::WeightLayoutTiling2D(
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    int vector_width)
    : WeightLayout(
          local_name_prefix + std::to_string(vector_width),
          proto_type, 2, 0.0f),
      vector_width_(vector_width) {}

CoordTransFunc WeightLayoutTiling2D::ToActual(const tvm::Tensor& /*X*/) const {
  return [&](const tvm::Array<tvm::Expr>& nominal_coord) {
    ORT_ENFORCE(nominal_coord.size() == 2);
    const auto& y = nominal_coord[0];
    const auto& x = nominal_coord[1];
    return tvm::Array<tvm::Expr>{
        x,
        y};
  };
}

CoordTransFunc WeightLayoutTiling2D::ToNominal(const tvm::Tensor& X) const {
  return [&](const tvm::Array<tvm::Expr>& actual_coord) {
    ORT_ENFORCE(actual_coord.size() == 2);
    ORT_ENFORCE(X->dtype == HalideIR::type_of<int8_t>() ||
                X->dtype == HalideIR::type_of<int16_t>());

    int tile_row = (sizeof(int32_t) * num_bits) / X->dtype.bits();
    int tile_col = ((vector_width_ * num_bits) / X->dtype.bits()) / tile_row;

    const auto& x = actual_coord[0];
    const auto& y = actual_coord[1];

    const int block_dimy = tile_row;
    const int block_dimx = tile_col;

    const auto& y0 = y % block_dimy;
    const auto& y1 = (y / block_dimy) % block_dimx;
    const auto& y2 = y / block_dimy / block_dimx;

    const auto& x0 = x % block_dimx;
    const auto& x1 = x / block_dimx;

    return tvm::Array<tvm::Expr>{
        y0 + y2 * block_dimx * block_dimy + x0 * block_dimy,
        y1 + x1 * block_dimx};
  };
}

tvm::Array<tvm::Expr> WeightLayoutTiling2D::ToActualShape(const tvm::Tensor& X) const {
  ORT_ENFORCE(X->dtype == HalideIR::type_of<int8_t>() ||
              X->dtype == HalideIR::type_of<int16_t>());

  auto pad_row = tvm::make_const(tvm::Int(32), (vector_width_ * num_bits) / X->dtype.bits());
  auto pad_col = tvm::make_const(tvm::Int(32), vector_width_ / sizeof(int32_t));

  auto new_shape0 = ((X->shape[1] + pad_col - 1) / pad_col) * pad_col;
  auto new_shape1 = ((X->shape[0] + pad_row - 1) / pad_row) * pad_row;

  tvm::Array<tvm::Expr>
      new_shape = {
          new_shape0,
          new_shape1};
  return new_shape;
}

std::vector<int64_t> WeightLayoutTiling2D::ToActualShape(const Tensor* X) const {
  ORT_ENFORCE(X != nullptr);
  ORT_ENFORCE(X->Shape().GetDims().size() == 2);

  int pad_row = vector_width_ / X->DataType()->Size();
  int pad_col = vector_width_ / sizeof(int32_t);

  auto old_shape = X->Shape().GetDims();
  auto new_shape0 = (old_shape[1] + pad_col - 1) / pad_col * pad_col;
  auto new_shape1 = ((old_shape[0] + pad_row - 1) / pad_row) * pad_row;

  std::vector<int64_t> new_shape = {
      new_shape0,
      new_shape1};

  return new_shape;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
