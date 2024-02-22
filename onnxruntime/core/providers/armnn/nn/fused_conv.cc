// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/armnn/armnn_execution_provider.h"
#include "core/providers/armnn/nn/conv.h"
#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/armnn_fwd.h"
#include "contrib_ops/cpu/fused_activation.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

class FusedConv final : public armnn_ep::Conv<float> {
 public:
  explicit FusedConv(const OpKernelInfo& info) : armnn_ep::Conv<float>(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("activation", &(this->activation_type)).IsOK());
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv);

}  // namespace armnn_ep
}  // namespace onnxruntime
