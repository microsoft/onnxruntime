// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace cann {

template <typename T>
class Conv final : public CannKernel {
 public:
  Conv(const OpKernelInfo& info) : CannKernel(info), conv_attrs_(info) {
    auto pads_size = conv_attrs_.pads.size();
    ORT_ENFORCE(pads_size % 2 == 0);
  }

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  ConvAttributes conv_attrs_;
};

}  // namespace cann
}  // namespace onnxruntime
