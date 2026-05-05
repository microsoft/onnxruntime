// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

class BitCast final : public OpKernel {
 public:
  explicit BitCast(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

}  // namespace onnxruntime
