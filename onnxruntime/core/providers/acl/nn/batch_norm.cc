// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"

#include <thread>
#include <mutex>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/acl/nn/batch_norm.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"

namespace onnxruntime {
namespace acl {

template <typename T>
thread_local std::map<OpKernel*, ACLNEBatchNorm> BatchNorm<T>::batchNormLayers;

template <typename T>
Status BatchNorm<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* S = context->Input<Tensor>(1);  // scale
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* M = context->Input<Tensor>(3);  // mean
  const Tensor* V = context->Input<Tensor>(4);  // var

  if (S->Shape().NumDimensions() > 1) {
    LOGS_DEFAULT(WARNING) << "ACL does not support scale with dimension greater then 1; defaulting to cpu implementation";
    return onnxruntime::BatchNorm<T>::Compute(context);
  }

  if (this->is_train_) {
    LOGS_DEFAULT(WARNING) << "ACL does not have batchnorm training support; defaulting to cpu implementation";
    return onnxruntime::BatchNorm<T>::Compute(context);
  }

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, S, B, M, V));

  LOGS_DEFAULT(VERBOSE) << "BatchNorm ACL:";
  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "params " << S->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << std::endl;

  const T* x_data = X->Data<T>();

  Tensor* Y = context->Output(0, X->Shape());

  T* y_data = Y->MutableData<T>();

  ACLNEBatchNorm* pBatchNorm;
  BatchNormLayersIterator it = batchNormLayers.find((OpKernel*)this);
  if (it == batchNormLayers.end()) {
    ACLNEBatchNorm tbatch_norm;
    tbatch_norm.in = std::make_shared<arm_compute::Tensor>();
    tbatch_norm.mean = std::make_shared<arm_compute::Tensor>();
    tbatch_norm.var = std::make_shared<arm_compute::Tensor>();
    tbatch_norm.b = std::make_shared<arm_compute::Tensor>();
    tbatch_norm.scale = std::make_shared<arm_compute::Tensor>();
    tbatch_norm.out = std::make_shared<arm_compute::Tensor>();

    auto layer = std::make_shared<arm_compute::NEBatchNormalizationLayer>();

    arm_compute::TensorShape in_x_shape;
    const TensorShape& x_shape = X->Shape();
    const auto& dims_vec = x_shape.GetDims();
    in_x_shape.set(3, onnxruntime::narrow<size_t>(dims_vec[0]));  // N
    in_x_shape.set(1, 1);                                         // H
    size_t W = 1;
    for (size_t i = 2; i < dims_vec.size(); ++i) {
      W *= narrow<size_t>(dims_vec[i]);
    }
    in_x_shape.set(0, W);                                         // W
    in_x_shape.set(2, onnxruntime::narrow<size_t>(dims_vec[1]));  // C

    tbatch_norm.in->allocator()->init(arm_compute::TensorInfo(in_x_shape, arm_compute::Format::F32));
    tbatch_norm.out->allocator()->init(arm_compute::TensorInfo(tbatch_norm.in->info()->tensor_shape(), arm_compute::Format::F32));

    tbatch_norm.scale->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(S->Shape()), arm_compute::Format::F32));
    tbatch_norm.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(B->Shape()), arm_compute::Format::F32));
    tbatch_norm.mean->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(M->Shape()), arm_compute::Format::F32));
    tbatch_norm.var->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(V->Shape()), arm_compute::Format::F32));

    layer->configure(tbatch_norm.in.get(), tbatch_norm.out.get(),
                     tbatch_norm.mean.get(), tbatch_norm.var.get(), B != nullptr ? tbatch_norm.b.get() : nullptr, S != nullptr ? tbatch_norm.scale.get() : nullptr,
                     epsilon_);  // no activation in onnx

    const T* scale_data = S->Data<T>();
    const T* b_data = B->Data<T>();
    const T* mean_data = M->Data<T>();
    const T* var_data = V->Data<T>();

    ACLImportMemory(tbatch_norm.mean->allocator(), (void*)mean_data, M->Shape().Size() * 4);
    ACLImportMemory(tbatch_norm.var->allocator(), (void*)var_data, V->Shape().Size() * 4);
    ACLImportMemory(tbatch_norm.b->allocator(), (void*)b_data, B->Shape().Size() * 4);
    ACLImportMemory(tbatch_norm.scale->allocator(), (void*)scale_data, S->Shape().Size() * 4);

    // allocate space for input tensor to accommodate paddings and strides
    tbatch_norm.in->allocator()->allocate();

    tbatch_norm.layer = std::move(layer);

    std::pair<BatchNormLayersIterator, bool> ret;
    ret = batchNormLayers.insert(std::pair<OpKernel*, ACLNEBatchNorm>((OpKernel*)this, tbatch_norm));
    pBatchNorm = &tbatch_norm;

  } else {
    pBatchNorm = &it->second;
  }

  if (X->Shape().Size() != 0 && pBatchNorm->in->info()->has_padding()) {
    importDataToTensor<T>(pBatchNorm->in.get(), x_data);
  } else {
    ACLImportMemory(pBatchNorm->in->allocator(), (void*)x_data, X->Shape().Size() * 4);
  }

  if (Y->Shape().Size() != 0 && pBatchNorm->out->info()->has_padding()) {
    pBatchNorm->out->allocator()->allocate();
  } else {
    ACLImportMemory(pBatchNorm->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);
  }

  pBatchNorm->layer->run();

  if (Y->Shape().Size() != 0 && pBatchNorm->out->info()->has_padding()) {
    importDataFromTensor<T>(pBatchNorm->out.get(), y_data);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    BatchNormalization,
    kOnnxDomain,
    7, 9,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    BatchNorm<float>);

}  // namespace acl
}  // namespace onnxruntime
