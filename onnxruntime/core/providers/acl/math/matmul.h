// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_execution_provider.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/IOperator.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"

namespace onnxruntime {
namespace acl {

Status ValidateMatMul(const onnxruntime::Node& node);

class MatMul : public OpKernel {
 public:
  explicit MatMul(const OpKernelInfo& info);

  Status PrePack(const Tensor&, int, AllocatorPtr,
          bool& is_packed, PrePackedWeights*) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>&,
          int, bool&) override;

  Status Compute(OpKernelContext* context) const override;

 protected:
  ACLExecutionProvider* provider_;
  std::shared_ptr<arm_compute::NEPermute> a_permute;
  std::shared_ptr<arm_compute::NEPermute> b_permute;
  std::shared_ptr<arm_compute::experimental::IOperator> layer;

  arm_compute::MemoryGroup memory_group;
  arm_compute::ITensorPack run_pack;
  arm_compute::ITensorPack prep_pack;

  Workspace workspace;

  std::shared_ptr<arm_compute::Tensor> a;
  std::shared_ptr<arm_compute::Tensor> b;
  std::shared_ptr<arm_compute::Tensor> a_transposed;
  arm_compute::Tensor *b_transposed = nullptr;
  std::shared_ptr<arm_compute::Tensor> out;
  arm_compute::Tensor *pb;

  IAllocatorUniquePtr<void> pbRaw;
  TensorShape outShape;
};
}  // namespace acl
}  // namespace onnxruntime
