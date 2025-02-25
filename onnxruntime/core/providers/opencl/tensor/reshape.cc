// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/opencl/tensor/reshape.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME reshape_kernel_src
#include "opencl_generated/tensor/kernels/reshape.cl.inc"
}  // namespace

TensorShape InferReshapeOutputShape(
    const TensorShape& data_tensor_shape,        // Data tensor's shape.
    const gsl::span<const int64_t>& shape_span,  // Shape that data tensor reshape to.
    bool allow_zero) {
  TensorShapeVector shape_vector(shape_span.begin(), shape_span.end());
  ReshapeHelper helper(data_tensor_shape, shape_vector, allow_zero);
  return TensorShape(shape_vector);
}

TensorShape InferReshapeOutputShape(const Tensor* src, const Tensor* shape, int64_t allow_zero) {
  ORT_ENFORCE(shape != nullptr, "Cannot reshape to a null shape.");
  ORT_ENFORCE(shape->Shape().NumDimensions() == 1, "Shape must be an 1-D tensor.");
  ORT_ENFORCE(shape->Location().device.Type() == OrtDevice::CPU, "Shape must be on CPU.");

  return InferReshapeOutputShape(
      src->Shape(),
      shape->template DataAsSpan<int64_t>(),
      allow_zero);
}
class Reshape : public OpenCLKernel {
 public:
  Reshape(const OpKernelInfo& info) : OpenCLKernel(info),
                                      allow_zero_(info.GetAttrOrDefault("allowzero", static_cast<int64_t>(0))) {
    LoadProgram(reshape_kernel_src, reshape_kernel_src_len);
    LoadKernel("Noop");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t allow_zero_;
};

Status Reshape::Compute(OpKernelContext* context) const {
  // Copy the second input tensor into the shape vector
  const Tensor* data_tensor = context->Input<Tensor>(0);
  const Tensor* shape_tensor = context->Input<Tensor>(1);
  const auto target_shape = InferReshapeOutputShape(data_tensor, shape_tensor, allow_zero_);
  Tensor* output_tensor = context->Output(0, target_shape);

  if (!data_tensor) return Status(common::ONNXRUNTIME, common::FAIL, "Missing data tensor to be reshaped.");
  if (!shape_tensor) return Status(common::ONNXRUNTIME, common::FAIL, "Missing shape tensor for reshaping.");
  if (shape_tensor->Shape().NumDimensions() != 1) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL, "The shape tensor for reshaping must be a vector, but got ", shape_tensor->Shape(), ".");
  }
  if (shape_tensor->Location().device.Type() != OrtDevice::CPU) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Shape tensor must be on CPU.");
  }

  cl_mem src_buffer = CL_BUFFER_FROM_TENSOR(*data_tensor);
  cl_mem dst_buffer = CL_BUFFER_FROM_TENSOR(*output_tensor);
  // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
  if (src_buffer != dst_buffer) {
    ORT_RETURN_IF_CL_ERROR(clEnqueueCopyBuffer(exec_->GetCommandQueue(), src_buffer, dst_buffer, /*src_offset=*/0, /*dst_offset=*/0, data_tensor->SizeInBytes(), 0, nullptr, nullptr));
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    14, 18,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    13, 13,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);
}  // namespace opencl
}  // namespace onnxruntime
