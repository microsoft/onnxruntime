// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reshape.h"

namespace onnxruntime {
namespace cuda {

TensorShape InferReshapeOutputShape(const Tensor* src, const Tensor* shape, bool allow_zero) {
    ORT_ENFORCE(shape != nullptr, "Cannot reshape to a null shape.");
    ORT_ENFORCE(shape->Shape().NumDimensions() != 1, "Shape must be an 1-D tensor.");
    ORT_ENFORCE(shape->Location().device.Type() == OrtDevice::CPU, "Shape must be on CPU.");

    auto shape_span = shape->template DataAsSpan<int64_t>();
    TensorShapeVector shape_vector(shape_span.begin(), shape_span.end());
    ReshapeHelper helper(src->Shape(), shape_vector, allow_zero);
  return TensorShape(shape_vector);
}

Status FuncReshape(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* X,
    const Tensor* shape,
    const bool /*allow_zero*/,
    Tensor* Y) {
    if (!X) return Status(common::ONNXRUNTIME, common::FAIL, "Missing data tensor to be reshaped.");
    if (!shape) return Status(common::ONNXRUNTIME, common::FAIL, "Missing shape tensor for reshaping.");
    if (shape->Shape().NumDimensions() != 1) {
      return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL, "The shape tensor for reshaping must be a vector, but got ", shape->Shape(), ".");
    }
    if (shape->Location().device.Type() != OrtDevice::CPU) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Shape tensor must be on CPU.");
    }

    const void* src_data = X->DataRaw();
    void* dst_data = Y->MutableDataRaw();
    // If source and target pointers are not equal (non-inplace operation), we need to copy the data.
    if (src_data != dst_data) {
      ORT_ENFORCE(ctx->GetComputeStream());
      ORT_RETURN_IF_ERROR(cuda_kernel->CopyTensor(*X, *Y, *ctx->GetComputeStream()));
    }

    return Status::OK();
}

std::unique_ptr<Tensor> FuncReshape(
    const CudaKernel* cuda_kernel,
    OpKernelContext* ctx,
    const Tensor* X,
    const Tensor* shape,
    const bool allow_zero
) {
  // TODO(wechi): Study if Tensor can be created as view to existing tensor.
  // This feature can refine code for re-sharding and shape broadcasting.

  ORT_ENFORCE(X != nullptr, "Missing data tensor to be reshaped.");
  ORT_ENFORCE(shape != nullptr, "Missing shape tensor for reshaping.");
  ORT_ENFORCE(shape->Shape().NumDimensions() == 1, "The shape tensor for reshaping must be a vector, but got ", shape->Shape(), ".");
  ORT_ENFORCE(shape->Location().device.Type() == OrtDevice::CPU, "Shape tensor must be on CPU.");

  // Calculate output's shape.
  auto dst_shape = InferReshapeOutputShape(X, shape, allow_zero);

  // Pre-allocate output.
  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc).IsOK());
  auto Y = Tensor::Create(X->DataType(), dst_shape, alloc);

  // Do reshape. It's equivalent to memcpy.
  ORT_ENFORCE(FuncReshape(cuda_kernel, ctx, X, shape, allow_zero, Y.get()).IsOK());
  return Y;
}

ONNX_OPERATOR_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    19,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypesIRv9())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    14, 18,
    kCudaExecutionProvider,
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
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    5, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    1,
    4,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Reshape_1);

}  // namespace cuda
}  // namespace onnxruntime
