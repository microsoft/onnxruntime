// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/deform_conv_attributes.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class DeformConv final : public WebGpuKernel {
 public:
  DeformConv(const OpKernelInfo& info) : WebGpuKernel(info), attrs_(info) {}

  Status ComputeInternal(ComputeContext& context) const override;
  Status PrePackInternal(ComputeContextBase& context,
                         const Tensor& tensor,
                         int input_idx,
                         AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

 private:
  DeformConvAttributes attrs_;
  std::unique_ptr<Tensor> packed_weight_;
  TensorShape packed_weight_source_shape_;
};

}  // namespace webgpu
}  // namespace onnxruntime
