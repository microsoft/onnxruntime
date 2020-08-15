// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License

#ifdef RELU_ARMNN

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/providers/armnn/armnn_execution_provider.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

typedef std::map<OpKernel*, armnn::NetworkId>::iterator ReluLayersIterator;

template <typename T>
class Relu : public OpKernel {
 public:
  explicit Relu(const OpKernelInfo& info) : OpKernel(info) {
  	provider_ = (const_cast<ArmNNExecutionProvider*>(
        static_cast<const ArmNNExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Relu() {
  	Relu::reluLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

  static armnn::IRuntimePtr initRuntime(){
  	if (Relu::run)
  		return std::move(Relu::run);
    armnn::IRuntime::CreationOptions options;
    return std::move(armnn::IRuntime::Create(options));
	}

 private:
  static thread_local std::map<OpKernel*, armnn::NetworkId> reluLayers;
  ArmNNExecutionProvider* provider_;
  static armnn::IRuntimePtr run;
};

}  // namespace armnn_ep
}  // namespace onnxruntime

#endif
