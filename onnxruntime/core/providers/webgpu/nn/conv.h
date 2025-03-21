// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/optional.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

template <bool is_channels_last, bool is_fused = false>
class Conv : public WebGpuKernel {
 public:
  Conv(const OpKernelInfo& info) : WebGpuKernel(info), conv_attrs_(info), activation(Activation::None) {
  }
  Status ComputeInternal(ComputeContext& context) const override;
  TensorShape ComputeOutputShape(const TensorShape& input_shape, const TensorShape& weight_shape, std::vector<uint32_t> pads, std::vector<uint32_t> strides, std::vector<uint32_t> dilations) const;

 protected:
  ConvAttributes conv_attrs_;
  Activation activation_;
};

}  // namespace webgpu
}  // namespace onnxruntime
