// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/passes/weight_layout/weight_layout.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

// WeightLayoutTranspose2D for transposing a 2D weight
// [W, H] => [H, W]
class WeightLayoutTranspose2D : public WeightLayout {
 public:
  static const std::string GetKey(ONNX_NAMESPACE::TensorProto_DataType proto_type);

 public:
  WeightLayoutTranspose2D(ONNX_NAMESPACE::TensorProto_DataType proto_type);

  ~WeightLayoutTranspose2D() = default;

  CoordTransFunc ToNominal(const tvm::Tensor& X) const override;
  CoordTransFunc ToActual(const tvm::Tensor& X) const override;
  tvm::Array<tvm::Expr> ToActualShape(const tvm::Tensor& X) const override;
  std::vector<int64_t> ToActualShape(const Tensor* X) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeightLayoutTranspose2D);
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
