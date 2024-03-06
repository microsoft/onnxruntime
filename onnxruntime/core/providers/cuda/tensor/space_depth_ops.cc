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
    SpaceToDepth<LAYOUT_NCHW>);

#ifdef ENABLE_CUDA_NHWC_OPS
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    SpaceToDepth,
    kMSInternalNHWCDomain,
    1,
    12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    SpaceToDepth<LAYOUT_NHWC>);
#endif

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
    SpaceToDepth<LAYOUT_NCHW>);

#ifdef ENABLE_CUDA_NHWC_OPS
ONNX_OPERATOR_KERNEL_EX(
    SpaceToDepth,
    kMSInternalNHWCDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    SpaceToDepth<LAYOUT_NHWC>);
#endif

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
    DepthToSpace<LAYOUT_NCHW>);

#ifdef ENABLE_CUDA_NHWC_OPS
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DepthToSpace,
    kMSInternalNHWCDomain,
    1,
    10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    DepthToSpace<LAYOUT_NHWC>);
#endif

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
    DepthToSpace<LAYOUT_NCHW>);

#ifdef ENABLE_CUDA_NHWC_OPS
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DepthToSpace,
    kMSInternalNHWCDomain,
    11,
    12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    DepthToSpace<LAYOUT_NHWC>);
#endif

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
    DepthToSpace<LAYOUT_NCHW>);

#ifdef ENABLE_CUDA_NHWC_OPS
ONNX_OPERATOR_KERNEL_EX(
    DepthToSpace,
    kMSInternalNHWCDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T",
                        {DataTypeImpl::GetTensorType<float>(),
                         DataTypeImpl::GetTensorType<double>(),
                         DataTypeImpl::GetTensorType<MLFloat16>()}),
    DepthToSpace<LAYOUT_NHWC>);
#endif

static Status SpaceDepthOpCudaImpl(const cudaDeviceProp& prop,
                                   cudaStream_t stream,
                                   const cublasHandle_t cublas_handle,
                                   const Tensor& input, Tensor& output,
                                   const std::vector<size_t>& permutation,
                                   const TensorShape& virtual_input_shape,
                                   const TensorShape& virtual_output_shape) {
  return Transpose::DoTranspose(prop, stream, cublas_handle, permutation, input, output,
                                &virtual_input_shape, &virtual_output_shape);
}

template <bool Layout>
Status SpaceToDepth<Layout>::ComputeInternal(OpKernelContext* context) const {
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

  ORT_RETURN_IF_ERROR(
      InputValidationsAndOutputDimsCalc<Layout == LAYOUT_NHWC>(input,
                                                               batch,
                                                               input_depth, input_height, input_width,
                                                               output_depth, output_height, output_width,
                                                               true));

  // We use the "actual" output shape to construct the output tensor
  Tensor& output = (Layout == LAYOUT_NCHW)
                       ? *context->Output(0, {batch, output_depth, output_height, output_width})
                       : *context->Output(0, {batch, output_height, output_width, output_depth});

  TensorShape virtual_input_shape = (Layout == LAYOUT_NCHW)
                                        ? TensorShape{batch, input_depth, input_height / blocksize_,
                                                      blocksize_, input_width / blocksize_, blocksize_}
                                        : TensorShape{batch, input_height / blocksize_, blocksize_,
                                                      input_width / blocksize_, blocksize_, input_depth};

  // We will pass in the "virtual" output shape to be used by DoTranspose() in SpaceDepthOpCudaImpl(...)
  TensorShape virtual_output_shape = (Layout == LAYOUT_NCHW)
                                         ? TensorShape{batch, blocksize_, blocksize_, input_depth,
                                                       input_height / blocksize_, input_width / blocksize_}
                                         : TensorShape{batch, input_height / blocksize_, input_width / blocksize_,
                                                       blocksize_, blocksize_, input_depth};

  std::vector<size_t> permutation = (Layout == LAYOUT_NCHW)
                                        ? std::vector<size_t>{0, 3, 5, 1, 2, 4}
                                        : std::vector<size_t>{0, 1, 3, 2, 4, 5};

  ORT_RETURN_IF_ERROR(
      SpaceDepthOpCudaImpl(GetDeviceProp(), Stream(context), GetCublasHandle(context), input, output, permutation,
                           virtual_input_shape, virtual_output_shape));

  return Status::OK();
}

template <bool Layout>
Status DepthToSpace<Layout>::ComputeInternal(OpKernelContext* context) const {
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

  ORT_RETURN_IF_ERROR(
      InputValidationsAndOutputDimsCalc<Layout == LAYOUT_NHWC>(input,
                                                               batch,
                                                               input_depth, input_height, input_width,
                                                               output_depth, output_height, output_width,
                                                               false));

  // We use the "actual" output shape to construct the output tensor
  Tensor& output = (Layout == LAYOUT_NCHW)
                       ? *context->Output(0, {batch, output_depth, output_height, output_width})
                       : *context->Output(0, {batch, output_height, output_width, output_depth});

  int64_t virtual_input_depth = input_depth / blocksize_ / blocksize_;
  TensorShape virtual_input_shape;

  // cdr only here!
  if (is_dcr_) {
    virtual_input_shape = (Layout == LAYOUT_NCHW)
                              ? TensorShape{batch, blocksize_, blocksize_,
                                            virtual_input_depth, input_height, input_width}
                              : TensorShape{batch, input_height, input_width,
                                            blocksize_, blocksize_, virtual_input_depth};
  } else {
    virtual_input_shape = (Layout == LAYOUT_NCHW)
                              ? TensorShape{batch, virtual_input_depth, blocksize_,
                                            blocksize_, input_height, input_width}
                              : TensorShape{batch, input_height, input_width,
                                            virtual_input_depth, blocksize_, blocksize_};
  }

  // We will pass in the "virtual" output shape to be used by DoTranspose() in SpaceDepthOpCudaImpl(...)
  TensorShape virtual_output_shape = (Layout == LAYOUT_NCHW)
                                         ? TensorShape{batch, virtual_input_depth, input_height,
                                                       blocksize_, input_width, blocksize_}
                                         : TensorShape{batch, input_height, blocksize_,
                                                       input_width, blocksize_, virtual_input_depth};

  std::vector<size_t> permutation;

  if (is_dcr_) {
    permutation = (Layout == LAYOUT_NCHW)
                      ? std::vector<size_t>({0, 3, 4, 1, 5, 2})
                      : std::vector<size_t>({0, 1, 3, 2, 4, 5});

  } else {
    permutation = std::vector<size_t>({0, 1, 4, 2, 5, 3});
  }

  ORT_RETURN_IF_ERROR(SpaceDepthOpCudaImpl(GetDeviceProp(), Stream(context), GetCublasHandle(context), input, output,
                                           permutation, virtual_input_shape, virtual_output_shape));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
