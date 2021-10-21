// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"

namespace onnxruntime {
namespace contrib {

// Record the event ID stored in the input tensor.
void record_event_in_tensor(const Tensor& event_id_tensor);

class RecordEvent final : public OpKernel {
public:
  RecordEvent(const OpKernelInfo& info) : OpKernel(info) { }
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime