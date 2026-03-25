// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/unsqueeze.h"

namespace onnxruntime {
namespace cuda {

namespace {

// PLUGIN BUILD ADAPTATION: PrepareCompute() is inherited from UnsqueezeBase
// in the non-plugin build, but the base class cannot be used in plugin builds
// because it depends on core/framework/op_kernel.h internals. This standalone
// function reimplements the same axes-parsing and output-shape computation.
Status PrepareComputeForPlugin(OpKernelContext* ctx, UnsqueezeBase::Prepare& p, const TensorShapeVector& axes_attr) {
  const auto* input = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input != nullptr);
  auto& input_tensor = *input;

  TensorShapeVector axes;
  size_t num_inputs = static_cast<size_t>(ctx->InputCount());
  if (num_inputs == 2) {
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 0 ||
                    axes_tensor->Shape().NumDimensions() == 1,
                "An axes tensor must be a scalar or a 1-D tensor.");
    auto data_span = axes_tensor->DataAsSpan<int64_t>();
    axes.assign(data_span.begin(), data_span.end());
  } else {
    axes.assign(axes_attr.begin(), axes_attr.end());
  }

  TensorShapeVector output_dims(axes.size() + input_tensor.Shape().NumDimensions(), 0);
  for (int64_t axis : axes) {
    axis = HandleNegativeAxis(axis, onnxruntime::narrow<int64_t>(output_dims.size()));
    if (axis < 0 || axis >= static_cast<int64_t>(output_dims.size())) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has an out of range axis");
    }

    auto axis_index = onnxruntime::narrow<size_t>(axis);
    if (output_dims[axis_index] != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "'axes' has a duplicate axis");
    }
    output_dims[axis_index] = 1;
  }

  auto begin = input_tensor.Shape().GetDims().begin();
  for (auto& axis_size : output_dims) {
    if (axis_size == 0) {
      axis_size = *begin++;
    }
  }

  TensorShape output_shape(output_dims);
  p.output_tensor = ctx->Output(0, output_shape);
  ORT_ENFORCE(p.output_tensor != nullptr);
  p.input_tensor = &input_tensor;
  return Status::OK();
}

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    1, 10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Unsqueeze);

// explicitly support negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    11, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Unsqueeze);

// axes is input instead of attribute, support bfloat16
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    13, 20,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Unsqueeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    21, 22,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Unsqueeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    23, 23,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Unsqueeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    24, 24,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Unsqueeze);

ONNX_OPERATOR_KERNEL_EX(
    Unsqueeze,
    kOnnxDomain,
    25,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Unsqueeze);

Status Unsqueeze::ComputeInternal(OpKernelContext* ctx) const {
  Prepare p;
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  ORT_RETURN_IF_ERROR(PrepareComputeForPlugin(ctx, p, axes_));
#else
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, p));
#endif

  const void* input = p.input_tensor->DataRaw();
  void* output = p.output_tensor->MutableDataRaw();
  if (input == output)
    return Status::OK();

  auto count = p.input_tensor->Shape().Size();
  auto element_bytes = p.input_tensor->DataType()->Size();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output, input, count * element_bytes, cudaMemcpyDeviceToDevice, Stream(ctx)));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
