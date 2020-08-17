// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/batch_norm.h"
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
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"

namespace onnxruntime {
namespace acl {

typedef struct {
  std::shared_ptr<arm_compute::NEBatchNormalizationLayer> layer;
  std::shared_ptr<arm_compute::Tensor> in, out;
  std::shared_ptr<arm_compute::Tensor> scale, b, mean, var;
} ACLNEBatchNorm;

typedef std::map<OpKernel*, ACLNEBatchNorm>::iterator BatchNormLayersIterator;

template <typename T>
class BatchNorm final : public OpKernel {
 public:
  explicit BatchNorm(const OpKernelInfo& info) : OpKernel(info) {
    auto st = info.GetAttr<float>("epsilon", &epsilon_);
    ORT_ENFORCE(st.IsOK(), st.ErrorMessage());

    provider_ = (const_cast<ACLExecutionProvider*>(
        static_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~BatchNorm() {
	batchNormLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
   float epsilon_;

 private:
  ACLExecutionProvider* provider_;
  static thread_local std::map<OpKernel*, ACLNEBatchNorm> batchNormLayers;
};



}  // namespace acl
}  // namespace onnxruntime
