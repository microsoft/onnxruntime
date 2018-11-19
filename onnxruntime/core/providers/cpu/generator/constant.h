// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class ConstantLike final : public OpKernel {
 public:
  ConstantLike(const OpKernelInfo& info) : OpKernel(info) {
    int64_t dtype = info.GetAttrOrDefault<int64_t>("dtype", 0 /*default_value*/);
    dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);

    shape_ = info.GetAttrsOrDefault<int64_t>("shape");
    ONNXRUNTIME_ENFORCE(info.GetAttr<float>("value", &value_).IsOK());
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  ONNX_NAMESPACE::TensorProto::DataType dtype_;
  std::vector<int64_t> shape_;
  float value_;
};

}  // namespace onnxruntime
