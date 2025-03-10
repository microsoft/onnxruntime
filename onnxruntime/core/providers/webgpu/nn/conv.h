// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

template <bool is_channels_last>
class Conv final : public WebGpuKernel {
 public:
  Conv(const OpKernelInfo& info) : WebGpuKernel(info), conv_attrs_(info) {
  }
  Status ComputeInternal(ComputeContext& context) const override;
  TensorShape ComputeOutputShape(const TensorShape& input_shape, const TensorShape& weight_shape) const;
 private:
  ConvAttributes conv_attrs_;
};

}  // namespace webgpu
}  // namespace onnxruntime
