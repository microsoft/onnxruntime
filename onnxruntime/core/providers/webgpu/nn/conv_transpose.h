// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/common.h"

#include "core/providers/cpu/nn/conv_transpose_attributes.h"
#include "core/providers/webgpu/webgpu_kernel.h"
namespace onnxruntime {
namespace webgpu {

template <bool is_channels_last>
class ConvTranspose final : public WebgpuKernel {
 public:
  ConvTranspose(const OpKernelInfo& info) : WebGpuKernel(info), conv_transpose_attrs_(info) {
  }
  Status ComputeInternal(ComputeContext& context);
  TensorShape ComputeOutputShape(const TensorShape& input_shape, const TensorShape& weight_shape, std::vector<uint32_t> pads, std::vector<uint32_t> strides, std::vector<uint32_t> dilations) const;
  protected:
  ConvTransposeAttributes conv_transpose_attrs_;
};

} // namespace webgpu
}  // namespace onnxruntime
