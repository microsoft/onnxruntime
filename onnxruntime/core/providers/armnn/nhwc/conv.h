// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
#include "core/providers/armnn/armnn_execution_provider.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep{

typedef std::map<OpKernel*, armnn::NetworkId>::iterator ConvLayersIterator;

template <typename T>
class NHWCConv : public onnxruntime::Conv<T> {
 public:
  explicit NHWCConv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info), conv_attrs_(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        dynamic_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~NHWCConv() {
    NHWCConv::convLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
    if (NHWCConv::run)
      return std::move(NHWCConv::run);
    armnn::IRuntime::CreationOptions options;
    return std::move(armnn::IRuntime::Create(options));
  }

 protected:
  static thread_local std::map<OpKernel*, armnn::NetworkId> convLayers;
  ConvAttributes conv_attrs_;
  ArmNNExecutionProvider* provider_;
  static armnn::IRuntimePtr run;
  std::string activation_type;

};

}  // namespace armnn_ep
}  // namespace onnxruntime
