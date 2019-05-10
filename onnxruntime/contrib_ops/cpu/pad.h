// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/pad.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
struct Pad final : public OpKernel, public PadBase {
  Pad(const OpKernelInfo& info) : OpKernel(info), PadBase(info, true) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime