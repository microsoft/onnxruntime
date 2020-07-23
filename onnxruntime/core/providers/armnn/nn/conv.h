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
class Conv : public onnxruntime::Conv<T> {
 public:
  explicit Conv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info), conv_attrs_(info) {
    provider_ = (const_cast<ArmNNExecutionProvider*>(
        static_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Conv() {
  	Conv::convLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
  	if (Conv::run)
  		return std::move(Conv::run);
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
