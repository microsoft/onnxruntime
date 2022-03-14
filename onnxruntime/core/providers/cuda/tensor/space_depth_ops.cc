// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "space_depth_ops.h"
#include "core/providers/cuda/tensor/transpose.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    SpaceToDepth,
    kOnnxDomain,
    1,
    12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    SpaceToDepth);

ONNX_OPERATOR_KERNEL_EX(
    SpaceToDepth,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    SpaceToDepth);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DepthToSpace,
    kOnnxDomain,
    1,
    10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    DepthToSpace);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DepthToSpace,
    kOnnxDomain,
    11,
    12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    DepthToSpace);

ONNX_OPERATOR_KERNEL_EX(
    DepthToSpace,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    DepthToSpace);

static Status SpaceDepthOpCudaImpl(const cudaDeviceProp& prop,
                                   cudaStream_t stream,
                                   const cublasHandle_t cublas_handle,
                                   const Tensor& input, Tensor& output,
                                   const std::vector<size_t>& permutation,
                                   const int64_t batch_size,
                                   const int64_t in_dim1, const int64_t in_dim2, const int64_t in_dim3,
                                   const int64_t in_dim4, const int64_t in_dim5) {
  TensorShape virtual_input_shape{batch_size, in_dim1, in_dim2, in_dim3, in_dim4, in_dim5};
  return Transpose::DoTranspose(prop, stream, cublas_handle, permutation, input, output, &virtual_input_shape);
}

Status SpaceToDepth::ComputeInternal(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *tensor_pointer;

  int64_t batch = -1;

  int64_t input_depth = -1;
  int64_t input_height = -1;
  int64_t input_width = -1;

  int64_t output_depth = -1;
  int64_t output_height = -1;
  int64_t output_width = -1;

  ORT_RETURN_IF_ERROR(InputValidationsAndOutputDimsCalc(input,
                                                        batch,
                                                        input_depth, input_height, input_width,
                                                        output_depth, output_height, output_width,
                                                        true));

  // We use the "virtual" output shape to construct the output tensor
  Tensor& output = *context->Output(0,
                                    {batch, blocksize_, blocksize_, input_depth, input_height / blocksize_, input_width / blocksize_});

  std::vector<size_t> permutation = {0, 3, 5, 1, 2, 4};

  ORT_RETURN_IF_ERROR(SpaceDepthOpCudaImpl(GetDeviceProp(), Stream(), CublasHandle(), input, output, permutation, batch,
                                           input_depth, input_height / blocksize_, blocksize_, input_width / blocksize_, blocksize_));

  // Reshape to "actual" output shape
  output.Reshape({batch, output_depth, output_height, output_width});

  return Status::OK();
}

Status DepthToSpace::ComputeInternal(OpKernelContext* context) const {
  const auto* tensor_pointer = context->Input<Tensor>(0);
  if (tensor_pointer == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& input = *tensor_pointer;

  int64_t batch = -1;

  int64_t input_depth = -1;
  int64_t input_height = -1;
  int64_t input_width = -1;

  int64_t output_depth = -1;
  int64_t output_height = -1;
  int64_t output_width = -1;

  ORT_RETURN_IF_ERROR(InputValidationsAndOutputDimsCalc(input,
                                                        batch,
                                                        input_depth, input_height, input_width,
                                                        output_depth, output_height, output_width,
                                                        false));

  // We use the "virtual" output shape to construct the output tensor
  Tensor& output = *context->Output(0,
                                    {batch, input_depth / blocksize_ / blocksize_, input_height, blocksize_, input_width, blocksize_});

  std::vector<size_t> permutation;
  permutation.reserve(6);
  permutation.push_back(0);

  if (is_dcr_) {
    permutation.push_back(3);
    permutation.push_back(4);
    permutation.push_back(1);
    permutation.push_back(5);
    permutation.push_back(2);

  } else {
    permutation.push_back(1);
    permutation.push_back(4);
    permutation.push_back(2);
    permutation.push_back(5);
    permutation.push_back(3);
  }

  int64_t dim1 = is_dcr_ ? blocksize_ : input_depth / blocksize_ / blocksize_;
  int64_t dim3 = is_dcr_ ? input_depth / blocksize_ / blocksize_ : blocksize_;

  ORT_RETURN_IF_ERROR(SpaceDepthOpCudaImpl(GetDeviceProp(), Stream(), CublasHandle(), input, output,
                                           permutation,
                                           batch,
                                           dim1, blocksize_, dim3, input_height, input_width));

  // Reshape to "actual" output shape
  output.Reshape({batch, output_depth, output_height, output_width});

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
