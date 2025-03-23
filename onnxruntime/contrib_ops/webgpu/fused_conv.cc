// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/conv.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {
using onnxruntime::webgpu::Conv;
class FusedConv final : public Conv<false, true> {
 public:
  FusedConv(const OpKernelInfo& info) : Conv<false, true>(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }
};

ONNX_OPERATOR_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", onnxruntime::webgpu::WebGpuSupportedFloatTypes()),
    FusedConv);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
