// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
//#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cpu/nn/conv.h"
#include "core/providers/cpu/nn/pool.h"
#include "contrib_ops/cpu/fused_activation.h"

#include "core/framework/op_kernel.h"
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
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

namespace onnxruntime {
namespace acl {

typedef struct
{
  std::shared_ptr<arm_compute::NEPermute> layer;
  std::shared_ptr<arm_compute::MemoryManagerOnDemand> mm_layer;
  std::shared_ptr<arm_compute::Tensor> in;
  std::shared_ptr<arm_compute::Tensor> out;
} ACLNEPermute;

typedef struct
{
  std::shared_ptr<arm_compute::IFunction> layer;
  std::shared_ptr<arm_compute::MemoryManagerOnDemand> mm_layer;
  std::shared_ptr<arm_compute::Tensor> in;
  std::shared_ptr<arm_compute::Tensor> k;
  std::shared_ptr<arm_compute::Tensor> b;
  std::shared_ptr<arm_compute::Tensor> out;
  bool isDepthwiseCPU;
} ACLNEConv;

typedef struct {
  std::shared_ptr<arm_compute::NEPoolingLayer> layer;
  std::shared_ptr<arm_compute::Tensor> in;
  std::shared_ptr<arm_compute::Tensor> out;
} ACLNEPool;

typedef std::map<OpKernel*, ACLNEPool>::iterator PoolLayersIterator;

template <typename T>
class ReorderInput : public OpKernel {
 public:
  ReorderInput(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReorderOutput : public OpKernel {
 public:
  ReorderOutput(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t channels_;
};

typedef std::map<OpKernel*, ACLNEConv>::iterator ConvLayersIterator;

template <typename T>
class NhwcConv : public onnxruntime::Conv<T> {
 public:
  explicit NhwcConv(const OpKernelInfo& info) : onnxruntime::Conv<T>(info), conv_attrs_(info) {

    provider_ = (const_cast<ACLExecutionProvider*>(
        dynamic_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));
  }

  ~NhwcConv() {
    NhwcConv::convLayers.erase(this);
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  static thread_local std::map<OpKernel*, ACLNEConv> convLayers;
  ConvAttributes conv_attrs_;
  ACLExecutionProvider* provider_;
  std::string activation_type;

  arm_compute::TensorShape ACLReshapeWeightsDepthwise(arm_compute::Tensor* kernel) const;
};

template <typename T>
class NhwcPoolBase : public PoolBase {
 public:
  NhwcPoolBase(const OpKernelInfo& info) : PoolBase(info) {
    if (!pool_attrs_.global_pooling)
      ORT_ENFORCE(pool_attrs_.kernel_shape.size() == 2, "kernel_shape num_dims is not compatible with X num_dims.");
  }

  Status NhwcPool(OpKernelContext* context, MLAS_POOLING_KIND kind) const;

private:
  static thread_local std::map<OpKernel*, ACLNEPool> poolLayers;
  ACLExecutionProvider* provider_;
};

template <typename T>
class NhwcMaxPool : public OpKernel, public NhwcPoolBase<T> {
 public:
  NhwcMaxPool(const OpKernelInfo& info) : OpKernel(info), NhwcPoolBase<T>(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class NhwcAveragePool : public OpKernel, public NhwcPoolBase<T> {
 public:
  NhwcAveragePool(const OpKernelInfo& info) : OpKernel(info), NhwcPoolBase<T>(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace acl
}  // namespace onnxruntime
