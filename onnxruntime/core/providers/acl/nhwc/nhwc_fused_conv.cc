// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/acl/nhwc/nhwc_ops.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"
#include "core/providers/acl/acl_execution_provider.h"
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace acl {

class NhwcFusedConv final : public NhwcConv<float> {
 public:
  explicit NhwcFusedConv(const OpKernelInfo& info) : NhwcConv<float>(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &(NhwcConv::activation_type)).IsOK());
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSNhwcDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NhwcFusedConv);

}  // namespace acl
}  // namespace onnxruntime
