// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/codegen/common/registry.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

using CoordTransFunc = std::function<tvm::Array<tvm::Expr>(const tvm::Array<tvm::Expr>&)>;

// WeightLayout is data layout transformer for weight/initializer
class WeightLayout {
 public:
  // Static function to return unique string as a key
  static const std::string GetKey(
      const std::string& name,
      ONNX_NAMESPACE::TensorProto_DataType proto_type,
      int input_dim,
      float pad_zero);

 public:
  WeightLayout(
      const std::string& name,
      ONNX_NAMESPACE::TensorProto_DataType proto_type,
      int input_dim,
      float pad_zero);

  virtual ~WeightLayout() = default;

  // Return a CoordTransFunc from actual (transformed) coordinate to normial (original) coordinate
  virtual CoordTransFunc ToNominal(const tvm::Tensor& X) const = 0;

  // Return a CoordTransFunc from normial (original) coordinate to actual (transformed) coordinate
  virtual CoordTransFunc ToActual(const tvm::Tensor& X) const = 0;

  // Return actual (transformed) shape in tvm::Array (tvm_codegen)
  virtual tvm::Array<tvm::Expr> ToActualShape(const tvm::Tensor& X) const = 0;

  // Return actual (transformed) shape in vector<int64_t> (ort)
  virtual std::vector<int64_t> ToActualShape(const Tensor* X) const = 0;

  // Create Layout Marshalling op in outputs
  void CreateLayoutMarshallingTVMOp(tvm::Array<tvm::Tensor>& inputs,
                                    tvm::Array<tvm::Tensor>& outputs) const;

  // Layout name
  const std::string& Name() const;

 protected:
  std::string name_;
  ONNX_NAMESPACE::TensorProto_DataType proto_type_;
  int input_dim_;
  float pad_zero_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WeightLayout);
};

// Weight Layout Registry is a registry holds all WeightLayout
using WeightLayoutRegistry = codegen::RegistryBase<WeightLayout>;

}  // namespace tvm_codegen
}  // namespace onnxruntime
