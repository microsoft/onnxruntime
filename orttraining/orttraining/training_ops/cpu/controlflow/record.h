// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "event_pool.h"

namespace onnxruntime {
namespace contrib {

class RecordEvent final : public OpKernel {
public:
  RecordEvent(const OpKernelInfo& info) : OpKernel(info) { }
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime