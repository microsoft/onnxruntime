// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/slice_grad.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/tensor/slice_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    SliceGrad,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(1)
        .InputMemoryType<OrtMemTypeCPUInput>(2)
        .InputMemoryType<OrtMemTypeCPUInput>(3)
        .InputMemoryType<OrtMemTypeCPUInput>(4)
        .InputMemoryType<OrtMemTypeCPUInput>(5)
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    SliceGrad);

Tensor* GetOutputGradientTensor(OpKernelContext* ctx) {
  const Tensor& shape = *ctx->Input<Tensor>(1);
  const TensorShape data_shape(shape.template Data<int64_t>(), shape.Shape().Size());
  return ctx->Output(0, data_shape);
}

const Tensor* SliceGrad::GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const {
  // The gradient computation logic is same as slice op except the assignment from input tensor to output tensor is
  // reversed, hence, the input tensor for slice op code (when used for gradient computation) would be the output
  // tensor for gradient op that will have the same shape as the input tensor for slice op when used for slicing and
  // not gradient computation.
  return GetOutputGradientTensor(ctx);
}

void SliceGrad::FillInputVectors(OpKernelContext* ctx, std::vector<int64_t>& input_starts,
                                 std::vector<int64_t>& input_ends, std::vector<int64_t>& input_axes,
                                 std::vector<int64_t>& input_steps) const {
  FillVectorsFromInput(*ctx->Input<Tensor>(2), *ctx->Input<Tensor>(3), ctx->Input<Tensor>(4),
                       ctx->Input<Tensor>(5), input_starts, input_ends, input_axes, input_steps);
}

Status SliceGrad::CallSliceImp(size_t element_size, size_t dimension_count, const TArray<int64_t>& starts_buffer,
                               const TArray<int64_t>& steps_buffer, const TArray<int64_t>& input_strides,
                               const TArray<fast_divmod>& output_strides, OpKernelContext* ctx,
                               const TensorShape& output_shape) const {
  Tensor* gradient_out_tensor = GetOutputGradientTensor(ctx);
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(gradient_out_tensor->MutableDataRaw(), 0, gradient_out_tensor->SizeInBytes(), Stream()));
  return SliceImplGrad(Stream(),
                       element_size,
                       gsl::narrow_cast<int32_t>(dimension_count),
                       starts_buffer,
                       steps_buffer,
                       input_strides,
                       output_strides,
                       ctx->Input<Tensor>(0)->DataRaw(),
                       gradient_out_tensor->MutableDataRaw(),
                       output_shape.Size());
}

}  // namespace cuda
}  // namespace onnxruntime
