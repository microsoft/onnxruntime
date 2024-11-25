// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv.h"
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
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"

namespace onnxruntime {
namespace acl {

Status ValidateConv(const onnxruntime::Node& node);

class Conv : public onnxruntime::OpKernel {
 public:
  explicit Conv(const OpKernelInfo& info);

  Status PrePack(const Tensor&, int, AllocatorPtr,
                 bool& is_packed, PrePackedWeights*) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>&,
                                   int, bool&) override;

  Status Compute(OpKernelContext* context) const override;

 protected:
  ConvAttributes conv_attrs_;
  ACLExecutionProvider* provider_;
  std::string activation_type;

  std::shared_ptr<arm_compute::IFunction> depthwise_layer;

  std::shared_ptr<arm_compute::experimental::IOperator> conv_layer;
  arm_compute::MemoryGroup memory_group;
  arm_compute::ITensorPack run_pack;
  arm_compute::ITensorPack prep_pack;

  Workspace workspace;

  std::shared_ptr<arm_compute::Tensor> in;
  std::shared_ptr<arm_compute::Tensor> k;
  IAllocatorUniquePtr<void> pkRaw;
  std::shared_ptr<arm_compute::Tensor> b;
  std::shared_ptr<arm_compute::Tensor> out;
  TensorShape outShape;
  bool is_channels_last;
  bool isQuantized;
  bool isDepthwiseCPU;
  bool has_bias;

  arm_compute::TensorShape ACLReshapeWeightsDepthwise(arm_compute::Tensor* kernel) const;
};
}  // namespace acl
}  // namespace onnxruntime
