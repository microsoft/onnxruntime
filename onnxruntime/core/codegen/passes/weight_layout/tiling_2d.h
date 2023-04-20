// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/passes/weight_layout/weight_layout.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

/*
 * \class! WeightLayoutTiling2D
 * \breif! Transform 2D weight to 4D by tiling both dimension,
 *         this layout is used for tensorization.
 * [M, N] => [M/Tx, N/Ty, Tx, Ty]
 */

class WeightLayoutTiling2D : public WeightLayout {
 public:
  static const std::string GetKey(ONNX_NAMESPACE::TensorProto_DataType proto_type,
                                  int vector_width);

 public:
  WeightLayoutTiling2D(ONNX_NAMESPACE::TensorProto_DataType proto_type,
                       int vector_width);

  ~WeightLayoutTiling2D() = default;

  CoordTransFunc ToNominal(const tvm::Tensor& X) const override;
  CoordTransFunc ToActual(const tvm::Tensor& X) const override;
  tvm::Array<tvm::Expr> ToActualShape(const tvm::Tensor& X) const override;
  std::vector<int64_t> ToActualShape(const Tensor* X) const override;

 private:
  int vector_width_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeightLayoutTiling2D);
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
