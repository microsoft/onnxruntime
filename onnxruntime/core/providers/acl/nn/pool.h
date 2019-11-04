// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool.h"
#include "core/providers/acl/acl_execution_provider.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

namespace onnxruntime {
namespace acl {

typedef struct {
  std::shared_ptr<arm_compute::NEPoolingLayer> layer;
  std::shared_ptr<arm_compute::Tensor> in;
  std::shared_ptr<arm_compute::Tensor> out;
} ACLNEPool;

typedef std::map<OpKernel*, ACLNEPool>::iterator PoolLayersIterator;

template <typename T, typename PoolType>
class Pool final : public onnxruntime::Pool<T, PoolType> {
 public:
  explicit Pool(const OpKernelInfo& info) : onnxruntime::Pool<T, PoolType>(info) {
    provider_ = (const_cast<ACLExecutionProvider*>(
        dynamic_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~Pool() {
    poolLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  static thread_local std::map<OpKernel*, ACLNEPool> poolLayers;
  ACLExecutionProvider* provider_;
};
}  // namespace acl
}
