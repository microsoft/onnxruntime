// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class ConstantOfShapeBase {

 protected:
  ConstantOfShapeBase(const OpKernelInfo& info);

  Status PrepareCompute(OpKernelContext* ctx, Tensor** output_tensor) const;

  void* GetValuePtr() const { return p_value_; }

 private:
  union SizeBasedValue {
    int8_t int8_;
    int16_t int16_;
    int32_t int32_;
    int64_t int64_;
  } s_value_;
  void* p_value_;

  void SetValue(size_t size, void* value) {
    switch (size) {
      case sizeof(int8_t):
        s_value_.int8_ = *(reinterpret_cast<int8_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int8_));
        break;
      case sizeof(int16_t):
        s_value_.int16_ = *(reinterpret_cast<int16_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int16_));
        break;
      case sizeof(int32_t):
        s_value_.int32_ = *(reinterpret_cast<int32_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int32_));
        break;
      case sizeof(int64_t):
        s_value_.int64_ = *(reinterpret_cast<int64_t*>(value));
        p_value_ = reinterpret_cast<void*>(&(s_value_.int64_));
        break;
      default:
        ORT_THROW("Unsupported value attribute datatype with sizeof=: ", size);
        break;
    }
  }

  void SetValueFromTensorProto(const ONNX_NAMESPACE::TensorProto&);
};

class ConstantOfShape final : public ConstantOfShapeBase, public OpKernel {
 public:
  explicit ConstantOfShape(const OpKernelInfo& info) : ConstantOfShapeBase(info), OpKernel(info) {};

  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace onnxruntime
