// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/ordered_reductions.h"

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(
    OrderedRowReduce, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    OrderedRowReduce);

ONNX_OPERATOR_KERNEL_EX(
    OrderedMatrixReduce128, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    OrderedMatrixReduce128);

OrderedRowReduce::OrderedRowReduce(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t width;
  ORT_THROW_IF_ERROR(info.GetAttr("width", &width));
  ORT_ENFORCE(width >= 32 && width <= 4096 && (width & (width - 1)) == 0,
              "width must be a power of two in [32,4096]");
  const int64_t normalize = info.GetAttrOrDefault<int64_t>("normalize", 0);
  ORT_ENFORCE(normalize == 0 || normalize == 1, "normalize must be zero or one");
  width_ = static_cast<int>(width);
  normalize_ = normalize != 0;
}

Status OrderedRowReduce::ComputeInternal(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto& shape = input->Shape();
  ORT_RETURN_IF(shape.NumDimensions() == 0, "OrderedRowReduce input rank must be positive");
  ORT_RETURN_IF(shape.GetDims().back() != width_, "Last dimension must match width");
  const int64_t rows = shape.Size() / width_;
  auto* output = context->Output(0, TensorShape{rows, 1});
  if (rows == 0) {
    return Status::OK();
  }
  return LaunchOrderedRowReduce(Stream(context), input->Data<float>(), output->MutableData<float>(),
                                static_cast<size_t>(rows), width_, normalize_);
}

Status OrderedMatrixReduce128::ComputeInternal(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto& shape = input->Shape();
  const size_t rank = shape.NumDimensions();
  ORT_RETURN_IF(rank < 2 || shape[rank - 2] != 128 || shape[rank - 1] != 128,
                "OrderedMatrixReduce128 input must end in [128,128]");
  TensorShapeVector output_shape(shape.GetDims().begin(), shape.GetDims().end() - 1);
  auto* output = context->Output(0, output_shape);
  const int64_t matrices = shape.Size() / (128 * 128);
  if (matrices == 0) {
    return Status::OK();
  }
  return LaunchOrderedMatrixReduce128(Stream(context), input->Data<float>(), output->MutableData<float>(),
                                      static_cast<size_t>(matrices));
}

}  // namespace onnxruntime::contrib::cuda
