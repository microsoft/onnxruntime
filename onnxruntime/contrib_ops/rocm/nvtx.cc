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

static std::unordered_map<int64_t, roctx_range_id_t> nvtx_ids;


Status NvtxPush::ComputeInternal(OpKernelContext* context) const {
  if (correlationId_ > 0) {
    if (nvtx_ids.find(correlationId_) == nvtx_ids.end()) {
      auto id = roctxRangeStartA(label_.c_str());
      nvtx_ids[correlationId_] = id;
      std::cout << "executing nvtx push label " << label_.c_str() << " cid " << correlationId_ << " rid " << id << std::endl;
    }
  }

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
       
  auto X_type = X->DataType();
  const void* source = X->DataRaw(X_type);
  void* target = Y->MutableDataRaw(X_type);
  printf("%p %p\n", source, target);
  if (target != source) {
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(target, source, X->Shape().Size() * X->DataType()->Size(), hipMemcpyDeviceToDevice, Stream()));
  }

  return Status::OK();
}

Status NvtxPop::ComputeInternal(OpKernelContext* context) const {
  if (correlationId_ > 0) {
    auto got = nvtx_ids.find(correlationId_);
    if (got != nvtx_ids.end()) {
      auto id = nvtx_ids[correlationId_];
      std::cout << "executing nvtx pop label " << label_.c_str() << " cid " << correlationId_ << " rid " << id << std::endl;
      roctxRangeStop(id);
      nvtx_ids.erase(got);
    }
  }

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
       
  auto X_type = X->DataType();
  const void* source = X->DataRaw(X_type);
  void* target = Y->MutableDataRaw(X_type);
  printf("%p %p\n", source, target);
  if (target != source) {
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(target, source, X->Shape().Size() * X->DataType()->Size(), hipMemcpyDeviceToDevice, Stream()));
  }

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
