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

// Layout of the kernel tensor that ComputeInternal consumes. `OIHW` is the layout the
// Conv operator is defined with; the others are produced by PrePackInternal and are each
// understood by exactly one consumer, so the layout has to be tracked explicitly rather
// than inferred from which prepacked tensor happens to be present.
enum class KernelLayout {
  OIHW,  // No prepacked tensor -- the kernel is read straight from input 1.
  HWIO,  // Consumed by grouped conv, the 1x1/same_size MatMul path and Conv2dMM.
  OHWI,  // Consumed by the Im2ColMatMul path only.
};

template <bool is_channels_last, bool is_fused>
class Conv : public WebGpuKernel {
 public:
  Conv(const OpKernelInfo& info) : WebGpuKernel(info), conv_attrs_(info) {
    if (is_fused) {
      ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
    }
  }
  Status ComputeInternal(ComputeContext& context) const override;

  Status PrePackInternal(ComputeContextBase& context,
                         const Tensor& tensor,
                         int input_idx,
                         AllocatorPtr alloc,
                         /*out*/ bool& is_packed) override;

 protected:
  ConvAttributes conv_attrs_;
  Activation activation_;
  // Set by PrePackInternal; null when the kernel could not be prepacked (e.g. the weight
  // is not a constant initializer), in which case ComputeInternal reads input 1 instead.
  std::unique_ptr<Tensor> prepacked_kernel_;
  // Layout of the tensor ComputeInternal ends up consuming -- `prepacked_kernel_` when it
  // is set, otherwise input 1. Stays `OIHW` while `prepacked_kernel_` is null.
  KernelLayout kernel_layout_{KernelLayout::OIHW};
};

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm);

}  // namespace webgpu
}  // namespace onnxruntime
