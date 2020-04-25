// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <thread>
#include <chrono>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "event_pool.h"

namespace onnxruntime {
namespace contrib {

class WaitEvent final : public OpKernel {
public:
  WaitEvent(const OpKernelInfo& info) : OpKernel(info) { }
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime