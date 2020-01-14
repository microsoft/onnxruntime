// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
#include "core/providers/acl/acl_execution_provider.h"

// ACL
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"

namespace onnxruntime {
namespace acl {

typedef struct
{
  std::shared_ptr<arm_compute::IFunction> layer;
  std::shared_ptr<arm_compute::MemoryManagerOnDemand> mm_layer;
  std::shared_ptr<arm_compute::Tensor> in;
  std::shared_ptr<arm_compute::Tensor> k;
  std::shared_ptr<arm_compute::Tensor> b;
  std::shared_ptr<arm_compute::Tensor> out;
  bool isDeptwise;
} ACLNEConv;

typedef std::map<OpKernel*, ACLNEConv>::iterator ConvLayersIterator;

template <typename T>
class Conv final : public onnxruntime::Conv<T> {
 public:
  explicit Conv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info), conv_attrs_(info) {
    provider_ = (const_cast<ACLExecutionProvider*>(
        dynamic_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Conv() {
    Conv::convLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  static thread_local std::map<OpKernel*, ACLNEConv> convLayers;
  ConvAttributes conv_attrs_;
  ACLExecutionProvider* provider_;

  arm_compute::TensorShape ACLReshapeWeightsDepthwise(arm_compute::Tensor* kernel);
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
