// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/conv.h"
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace contrib {

class FusedConvFloat final : public ConvFloat {
 public:
  FusedConvFloat(const OpKernelInfo& info) : ConvFloat(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }
};

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    FusedConv,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConvFloat);

}  // namespace contrib
}  // namespace onnxruntime
