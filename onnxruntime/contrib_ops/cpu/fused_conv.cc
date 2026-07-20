// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/conv.h"
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace contrib {

class FusedConvFloat final : public Conv<float> {
 public:
  FusedConvFloat(const OpKernelInfo& info) : Conv<float>(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }
};

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    FusedConv,
    1,
    float,
    KernelDefBuilder()
        // If the fused "sum" input is available to the FusedConv node,
        // provide a hint to the allocation planner that it can re-use the
        // "sum"'s input as the output buffer of the FusedConv node.
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConvFloat);

#ifdef USE_KLEIDIAI
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    NhwcFusedConv,
    1,
    float,
    KernelDefBuilder()
        // Allow the optional "sum" input (index 3) to be reused as the output buffer (index 0),
        // consistent with the FusedConv kernel registration.
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConvFloat);
#endif

}  // namespace contrib
}  // namespace onnxruntime
