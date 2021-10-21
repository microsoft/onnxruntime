// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/activation/activations.h"
#include "core/providers/acl/acl_fwd.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"

namespace onnxruntime {
namespace acl {

template <typename T>
Status Relu<T>::Compute(OpKernelContext* context) const {
  arm_compute::Tensor in, out;
  arm_compute::NEActivationLayer layer;

  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());

  const T* src_data = X->template Data<T>();
  T* dst_data = Y->template MutableData<T>();

  in.allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape()), arm_compute::Format::F32));
  ACLImportMemory(in.allocator(), (void*)src_data, X->Shape().Size() * 4);

  out.allocator()->init(arm_compute::TensorInfo(ACLTensorShape(Y->Shape()), arm_compute::Format::F32));
  ACLImportMemory(out.allocator(), dst_data, Y->Shape().Size() * 4);

  layer.configure(&in, &out, arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::RELU));
  layer.run();

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Relu<float>);

}  // namespace acl
}  // namespace onnxruntime
