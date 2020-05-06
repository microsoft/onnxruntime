// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include <thread>
#include <mutex>

#include "core/providers/acl/nn/conv.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"
#include "core/providers/acl/acl_execution_provider.h"

namespace onnxruntime {
namespace acl{

template <typename T>
class FusedConv final : public acl::Conv<T> {
 public:
  explicit FusedConv(const OpKernelInfo& info) : acl::Conv<T>(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &(this->activation_type)).IsOK());
    // printf("fused\n");
  }
  // Status Compute(OpKernelContext* context) const override;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);

}  // namespace acl
}  // namespace onnxruntime
