// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/tensor/unsqueeze.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/tensor/unsqueeze.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME unsqueeze_kernel_src
#include "opencl_generated/tensor/kernels/unsqueeze.cl.inc"
}  // namespace

class Unsqueeze final : public OpenCLKernel, public UnsqueezeBase {
 public:
  Unsqueeze(const OpKernelInfo& info) : OpenCLKernel(info), UnsqueezeBase(info) {
    LoadProgram(unsqueeze_kernel_src, unsqueeze_kernel_src_len);
    LoadKernel("Nop");
  }
  Status Compute(OpKernelContext* context) const override;
};

Status Unsqueeze::Compute(OpKernelContext* ctx) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, p));
  const void* input = p.input_tensor->DataRaw();
  void* output = p.output_tensor->MutableDataRaw();
  if (input == output)
    return Status::OK();

  size_t count = p.input_tensor->Shape().Size();
  size_t element_bytes = p.input_tensor->DataType()->Size();
  size_t total_size = count * element_bytes;

  cl_int err = clEnqueueCopyBuffer(exec_->GetCommandQueue(), CL_BUFFER_FROM_TENSOR(*(p.input_tensor)), CL_BUFFER_FROM_TENSOR(*(p.output_tensor)), 0, 0, total_size, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to copy buffer");
  }
  clFinish(exec_->GetCommandQueue());

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    13, 20,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Unsqueeze)

}  // namespace opencl
}  // namespace onnxruntime
