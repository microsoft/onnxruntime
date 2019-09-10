// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/codegen/passes/weight_layout/weight_layout.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

// WeightLayoutVerticalStripe2D for making a 2D weight to 3D, by tiling the lowest (verteical) dimension
// [W, H] => [H/stripe, W, stripe]
class WeightLayoutVerticalStripe2D : public WeightLayout {
 public:
  static const std::string GetKey(
      ONNX_NAMESPACE::TensorProto_DataType proto_type,
      int stripe_width);

 public:
  WeightLayoutVerticalStripe2D(
      ONNX_NAMESPACE::TensorProto_DataType proto_type,
      int stripe_width);

  ~WeightLayoutVerticalStripe2D() = default;

  virtual CoordTransFunc ToNominal(const tvm::Tensor& X) const override;
  virtual CoordTransFunc ToActual(const tvm::Tensor& X) const override;
  tvm::Array<tvm::Expr> ToActualShape(const tvm::Tensor& X) const override;
  std::vector<int64_t> ToActualShape(const Tensor* X) const override;

 private:
  int stripe_width_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeightLayoutVerticalStripe2D);
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
