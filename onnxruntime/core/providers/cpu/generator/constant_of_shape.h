// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class ConstantOfShapeBase {
  union AttrValue {
    float fl_;
    MLFloat16 fl16_;
    double dbl_;
    int64_t i64_;
    uint64_t ui64_;
    AttrValue() : ui64_(0) {}

    float GetFloat() const {
      return fl_;
    }

    MLFloat16 GetFloat16() const {
      return fl16_;
    }

    double GetDouble() const {
      return dbl_;
    }

    template <class T>
    T GetFromSigned() const;

    template <class T>
    T GetFromUnsigned() const;

  };

 protected:
  ConstantOfShapeBase(const OpKernelInfo& info);
  AttrValue GetAttrValue() const { return value_; }
  ONNX_NAMESPACE::TensorProto_DataType GetTensorType() const { return tensor_type_; }

  Status PrepareCompute(OpKernelContext* ctx, Tensor** output_tensor) const;

 private:
  ONNX_NAMESPACE::TensorProto_DataType tensor_type_;
  AttrValue value_;

  void SetValue(const ONNX_NAMESPACE::TensorProto&);
};

class ConstantOfShape final : public ConstantOfShapeBase, public OpKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), OpKernel(info) {};

  Status Compute(OpKernelContext* ctx) const override;

 private:
  void DispatchTypeAndFillOutput(Tensor* output_tensor) const;
};

}  // namespace onnxruntime
