// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/reduction/reduction_ops.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_MS_KERNEL_TYPED(name, T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      name,                                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, MLFloat16)
REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, float)
REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, double)
REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, int32_t)

template <bool allow_multi_axes>
template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceKernel<allow_multi_axes>::ComputeImplEx(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  // override the attribute value with the input value for reduction_axes
  const Tensor* axes_tensor = ctx->Input<Tensor>(1);
  ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
  ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
  auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
  const auto* data = axes_tensor->template Data<int64_t>();
  std::vector<int64_t> axes(data, data + nDims);

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* Y = ctx->Output(0, X->Shape());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->template MutableData<T>(), X->template Data<T>(), X->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(ctx)));
    return Status::OK();
  }

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes,
                                       prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  const bool fast_reduction = fast_reduction_ && !ctx->GetUseDeterministicCompute();

  return ReduceComputeCore<T, ReduceTensorIndices>(Info().GetAllocator(OrtMemType::OrtMemTypeDefault), *X, prepare_reduce_metadata, *Y, cudnn_reduce_op, axes,
                                                   calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction, ctx->GetComputeStream());
}

template <>
template <>
Status ReduceKernel<true>::ComputeImplEx<int32_t, CUDNN_REDUCE_TENSOR_NO_INDICES>(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const {
  typedef typename ToCudaType<int32_t>::MappedType CudaT;

  const Tensor* X = ctx->Input<Tensor>(0);

  // override the attribute value with the input value for reduction_axes
  const Tensor* axes_tensor = ctx->Input<Tensor>(1);
  ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
  auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
  const auto* data = axes_tensor->template Data<int64_t>();
  std::vector<int64_t> axes(data, data + nDims);

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* Y = ctx->Output(0, X->Shape());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->template MutableData<int32_t>(), X->template Data<int32_t>(), X->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(ctx)));
    return Status::OK();
  }

  PrepareReduceMetadata prepare_reduce_metadata;

  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes,
                                       prepare_reduce_metadata));

  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);

  int64_t input_count = prepare_reduce_metadata.input_count;
  int64_t output_count = prepare_reduce_metadata.output_count;
  auto& input_dims_cudnn = prepare_reduce_metadata.input_dims_cudnn;
  auto& output_dims_cudnn = prepare_reduce_metadata.output_dims_cudnn;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(Y->Shape().Size() == 0);
    return Status::OK();
  }

  // cudnnReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
  if (input_count == output_count) {
    if (Y->template MutableData<int32_t>() != X->template Data<int32_t>()) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->template MutableData<int32_t>(), X->template Data<int32_t>(), input_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, Stream(ctx)));
    }
    return Status::OK();
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream(ctx)));

  size_t indices_bytes = 0;
  size_t workspace_bytes = 0;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  CudnnReduceDescriptor reduce_desc;

  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;
  IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count, ctx->GetComputeStream());
  Impl_Cast<CudaT, float>(Stream(ctx), reinterpret_cast<const CudaT*>(X->template Data<int32_t>()), temp_X.get(), X->Shape().Size());

  ORT_RETURN_IF_ERROR(reduce_desc.Set(cudnn_reduce_op, cudnn_type_X, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES));
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_cudnn, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_cudnn, cudnn_type_X));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(GetCudnnHandle(ctx), reduce_desc, input_tensor, output_tensor, &indices_bytes));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(GetCudnnHandle(ctx), reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  IAllocatorUniquePtr<uint32_t> indices_cuda = GetScratchBuffer<uint32_t>(indices_bytes, ctx->GetComputeStream());
  IAllocatorUniquePtr<CudaT> workspace_cuda = GetScratchBuffer<CudaT>(workspace_bytes, ctx->GetComputeStream());

  const auto one = Consts<float>::One;
  const auto zero = Consts<float>::Zero;
  auto temp_Y = GetScratchBuffer<float>(output_count, ctx->GetComputeStream());
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(GetCudnnHandle(ctx),
                                          reduce_desc,
                                          indices_cuda.get(),
                                          indices_bytes,
                                          workspace_cuda.get(),
                                          workspace_bytes,
                                          &one,
                                          input_tensor,
                                          temp_X.get(),
                                          &zero,
                                          output_tensor,
                                          temp_Y.get()));

  Impl_Cast<float, int32_t>(Stream(ctx), temp_Y.get(), Y->template MutableData<int32_t>(), output_count);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
