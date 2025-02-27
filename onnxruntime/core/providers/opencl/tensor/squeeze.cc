// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/tensor/squeeze.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/tensor/squeeze.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME squeeze_kernel_src
#include "opencl_generated/tensor/kernels/squeeze.cl.inc"
}  // namespace

class Squeeze final : public OpenCLKernel, public SqueezeBase {
 public:
  Squeeze(const OpKernelInfo& info) : OpenCLKernel(info), SqueezeBase(info) {
    LoadProgram(squeeze_kernel_src, squeeze_kernel_src_len);
    LoadKernel("Nop");
  }
  Status Compute(OpKernelContext* context) const override;
};

Status Squeeze::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();

  TensorShapeVector axes;
  size_t num_inputs = context->InputCount();
  if (num_inputs == 2) {  // axes is an input
    const Tensor* axes_tensor = context->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->Data<int64_t>();
    axes.assign(data, data + nDims);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  TensorShapeVector output_shape = ComputeOutputShape(X_shape, axes);

  Tensor* Y = context->Output(0, TensorShape(output_shape));
  if (X->DataRaw() == Y->MutableDataRaw())
    return Status::OK();
  size_t TotalSize = Y->SizeInBytes();
  if (TotalSize > 0) {
    cl_int err = clEnqueueCopyBuffer(exec_->GetCommandQueue(),
                                     CL_BUFFER_FROM_TENSOR(*X), CL_BUFFER_FROM_TENSOR(*Y),
                                     0, 0, TotalSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to copy buffer");
    }
    clFinish(exec_->GetCommandQueue());
  }
  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    13, 20,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Squeeze)

}  // namespace opencl
}  // namespace onnxruntime
