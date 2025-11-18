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

template <bool is_channels_last, bool is_fused>
class Conv : public WebGpuKernel {
 public:
  Conv(const OpKernelInfo& info) : WebGpuKernel(info), conv_attrs_(info) {
    if (is_fused) {
      ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
    }
    // Extract weight input name (input index 1) for caching
    const auto& input_defs = info.node().InputDefs();
    if (input_defs.size() > 1 && input_defs[1]->Exists()) {
      weight_name_ = input_defs[1]->Name();
    }
  }
  Status ComputeInternal(ComputeContext& context) const override;

 protected:
  ConvAttributes conv_attrs_;
  Activation activation_;
  std::string weight_name_;  // Name of weight input for cache key

  // Cached transformed weight pointer (set on first inference)
  mutable const Tensor* transformed_weight_ = nullptr;
  mutable bool weight_transform_attempted_ = false;

  // Get or create transformed weight (lazy transformation on first inference)
  Status GetTransformedWeight(ComputeContext& context, const Tensor* original_weight,
                              const Tensor*& transformed_weight) const;
};

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm);

}  // namespace webgpu
}  // namespace onnxruntime
