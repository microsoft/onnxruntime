// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/nvtx.h"
#include "core/providers/common.h"

#include "roctracer/roctx.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {


ONNX_OPERATOR_KERNEL_EX(
    NvtxPush,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    KernelDefBuilder().Alias(0, 0),
    NvtxPush);

ONNX_OPERATOR_KERNEL_EX(
    NvtxPop,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    KernelDefBuilder().Alias(0, 0),
    NvtxPop);

Status NvtxPush::ComputeInternal(OpKernelContext* context) const {
  std::cout << "executing nvtx push" << std::endl;
  roctxRangePushA("myrange");

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
       
  auto X_type = X->DataType();
  const void* source = X->DataRaw(X_type);
  void* target = Y->MutableDataRaw(X_type);
  printf("%p %p\n", source, target);
  if (target != source) {
    // ORT_THROW("target != source");
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(target, source, X->Shape().Size() * X->DataType()->Size(), hipMemcpyDeviceToDevice, Stream()));
  }

  return Status::OK();
}

Status NvtxPop::ComputeInternal(OpKernelContext* context) const {
  std::cout << "executing nvtx pop" << std::endl;
  roctxRangePop();

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
       
  auto X_type = X->DataType();
  const void* source = X->DataRaw(X_type);
  void* target = Y->MutableDataRaw(X_type);
  printf("%p %p\n", source, target);
  if (target != source) {
    // ORT_THROW("target != source");
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(target, source, X->Shape().Size() * X->DataType()->Size(), hipMemcpyDeviceToDevice, Stream()));
  }


  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
