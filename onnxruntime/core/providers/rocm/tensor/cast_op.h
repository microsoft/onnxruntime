// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/hip_common.h"

namespace onnxruntime {
namespace rocm {

template <typename SrcT>
class Cast final : public RocmKernel {
 public:
  Cast(const OpKernelInfo& info) : RocmKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

}  // namespace rocm
}  // namespace onnxruntime
