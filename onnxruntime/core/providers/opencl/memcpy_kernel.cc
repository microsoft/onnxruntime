// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_utils.h"
#include "opencl_data_transfer.h"

#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
namespace opencl {

class Memcpy final : public OpKernel {
 public:
  explicit Memcpy(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
  auto X_type = ctx->InputType(0);
  if (X_type->IsTensorType()) {
    const auto* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
    auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device, Y->Location().device);
    return gpu_data_transfer->CopyTensor(*X, *Y);
  } else {
    if (X_type->IsSparseTensorType()) {
      // TODO: support aysnc copy for sparse tensor
      return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
    } else if (X_type->IsTensorSequenceType()) {
      const TensorSeq* X = ctx->Input<TensorSeq>(0);
      ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor sequence is nullptr.");
      TensorSeq* Y = ctx->Output<TensorSeq>(0);
      ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor sequence.");
      return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
      } else {
        // If we are copying contents to CPU (op type is "MemcpyToHost"),
        // the allocator to use to allocate the buffers of the new tensors
        // in the sequence will be the allocator from the CPU EP
        return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
    }
    return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
  }
  }

};


ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

}  // namespace opencl
}  // namespace onnxruntime
