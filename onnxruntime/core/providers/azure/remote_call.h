// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace azure {

class RemoteCall : public OpKernel {
 public:
  RemoteCall(const OpKernelInfo& info) : OpKernel(info) {};
  common::Status Compute(OpKernelContext* context) const override;
};

}  // namespace azure
}  // namespace onnxruntime