// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/tensor/upsample.h"

namespace onnxruntime {
namespace webgpu {

class Resize : public Upsample {
 public:
  Resize(const OpKernelInfo& info) : Upsample(info) {
  }

  Status ComputeInternal(ComputeContext& context) const override {
    return Upsample::ComputeInternal(context);
  }
};

}  // namespace webgpu
}  // namespace onnxruntime
