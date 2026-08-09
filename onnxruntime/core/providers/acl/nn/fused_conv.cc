// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {
namespace acl {

class FusedConv final : public acl::Conv {
 public:
  explicit FusedConv(const OpKernelInfo& info) : acl::Conv(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &(this->activation_type)).IsOK());
    MLAS_ACTIVATION activation;
    activation.ActivationKind = MlasIdentityActivation;
    ORT_ENFORCE(GetFusedActivationAttr(info, activation).IsOK());
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv);

}  // namespace acl
}  // namespace onnxruntime
