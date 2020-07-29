// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
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
  const Tensor* S = context->Input<Tensor>(1);//scale
  const Tensor* B = context->Input<Tensor>(2);
  const Tensor* M = context->Input<Tensor>(3);//mean
  const Tensor* V = context->Input<Tensor>(4);//var

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, S, B, M, V));

  LOGS_DEFAULT(VERBOSE) << "BatchNorm ACL:";  
  LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << "params " << S->Shape().ToString().c_str();
  LOGS_DEFAULT(VERBOSE) << std::endl;

  const T* x_data = X->template Data<T>();

  Tensor* Y = context->Output(0, X->Shape());

  T* y_data = Y->template MutableData<T>();

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

    tbatch_norm.in->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape()), arm_compute::Format::F32));
    tbatch_norm.out->allocator()->init(arm_compute::TensorInfo(tbatch_norm.in->info()->tensor_shape(), arm_compute::Format::F32));

    tbatch_norm.scale->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(S->Shape()), arm_compute::Format::F32));
    tbatch_norm.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(B->Shape()), arm_compute::Format::F32));
    tbatch_norm.mean->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(M->Shape()), arm_compute::Format::F32));
    tbatch_norm.var->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(V->Shape()), arm_compute::Format::F32));

    layer->configure(tbatch_norm.in.get(), tbatch_norm.out.get(),
      tbatch_norm.mean.get(), tbatch_norm.var.get(), B != nullptr ? tbatch_norm.b.get() : nullptr, S != nullptr ? tbatch_norm.scale.get() : nullptr,
      epsilon_);//no activation in onnx

    const T* scale_data = S->template Data<T>();
    const T* b_data = B->template Data<T>();
    const T* mean_data = M->template Data<T>();
    const T* var_data = V->template Data<T>();

    ACLImportMemory(tbatch_norm.mean->allocator(), (void*)mean_data, M->Shape().Size() * 4);
    ACLImportMemory(tbatch_norm.var->allocator(), (void*)var_data, V->Shape().Size() * 4);
    ACLImportMemory(tbatch_norm.b->allocator(), (void*)b_data, B->Shape().Size() * 4);
    ACLImportMemory(tbatch_norm.scale->allocator(), (void*)scale_data, S->Shape().Size() * 4);

    // allocate space for input tensor to accomodate paddings and strides
    tbatch_norm.in->allocator()->allocate();

    tbatch_norm.layer = std::move(layer);

    std::pair<BatchNormLayersIterator, bool> ret;
    ret = batchNormLayers.insert(std::pair<OpKernel*, ACLNEBatchNorm>((OpKernel*)this, tbatch_norm));
    pBatchNorm = &tbatch_norm;

  } else {
    pBatchNorm = &it->second;
  }


  if(X->Shape().Size() != 0 && pBatchNorm->in->info()->has_padding() ){
    arm_compute::Window aclInpuWindow;
    aclInpuWindow.use_tensor_dimensions(pBatchNorm->in->info()->tensor_shape());

    arm_compute::Iterator aclInputIt(pBatchNorm->in.get(), aclInpuWindow);
    const unsigned int aclWidth = pBatchNorm->in->info()->dimension(0);
    const unsigned int aclHeight = pBatchNorm->in->info()->dimension(1);

    // copy input tensor into the larger buffer
    arm_compute::execute_window_loop(
      aclInpuWindow,
      [&](const arm_compute::Coordinates& co) {
        *reinterpret_cast<float*>(aclInputIt.ptr()) = x_data[co.z() * (aclWidth * aclHeight) + co.y() * aclHeight + co.x()];
      },
      aclInputIt);
  }else{
    ACLImportMemory(pBatchNorm->in->allocator(), (void*)x_data, X->Shape().Size() * 4);
  }


  if(Y->Shape().Size() != 0 && pBatchNorm->out->info()->has_padding() ){
    pBatchNorm->out->allocator()->allocate();
  } else {
    ACLImportMemory(pBatchNorm->out->allocator(), (void*)y_data, Y->Shape().Size() * 4);
  }

  pBatchNorm->layer->run();

  if(Y->Shape().Size() != 0 && pBatchNorm->out->info()->has_padding() ){
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
      .TypeConstraint("X", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("scale", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("B", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("mean", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("var", DataTypeImpl::GetTensorType<float>()),
    BatchNorm<float>);

}  // namespace acl
}  // namespace onnxruntime
