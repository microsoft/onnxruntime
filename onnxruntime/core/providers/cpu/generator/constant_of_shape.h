// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class ConstantOfShape final : public OpKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType tensor_type_;
  union Value {
    float fl_;
    MLFloat16 fl16_;
    double dbl_;
    int64_t i64_;
    uint64_t ui64_;
    Value() : ui64_(0) {}

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

  } value_;

  void SetValue(const ONNX_NAMESPACE::TensorProto&);

  void DispatchTypeAndFillOutput(Tensor* output_tensor) const;
};

}  // namespace onnxruntime
