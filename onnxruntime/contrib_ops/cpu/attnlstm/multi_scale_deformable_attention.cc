// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/attnlstm/multi_scale_deformable_attention.h"

#include "core/framework/op_kernel.h"

#include <memory>

namespace onnxruntime {
namespace contrib {

MultiScaleDeformableAttention::MultiScaleDeformableAttention(const OpKernelInfo& info) : OpKernel(info) {
}

Status MultiScaleDeformableAttention::Compute(_Inout_ OpKernelContext* context) const {
  return Status::OK();
}

ONNX_CPU_OPERATOR_MS_KERNEL(
    MultiScaleDeformableAttention,
    1,
    KernelDefBuilder().TypeConstraint(
        "T",
        {DataTypeImpl::GetTensorType<float>()}),
    MultiScaleDeformableAttention)

}  // namespace contrib
}  // namespace onnxruntime
