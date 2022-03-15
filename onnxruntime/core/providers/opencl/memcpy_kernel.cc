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
    const auto* X_type = ctx->InputType(0);
    if (X_type->IsTensorType()) {
      const auto* X = ctx->Input<Tensor>(0);
      Tensor& Y = ctx->RequiredOutput(0, X->Shape());
      return Info().GetDataTransferManager().CopyTensor(*X, Y, Info().GetKernelDef().ExecQueueId());
    }
    return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
  }
};

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

}  // namespace opencl
}  // namespace onnxruntime
