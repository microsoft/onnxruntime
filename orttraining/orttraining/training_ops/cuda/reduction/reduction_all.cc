// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/reduction/reduction_all.h"

#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_REDUCE_ALL_KERNEL_TYPED(Name, TIn, TOut)                                                                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                                           \
      Name,                                                                                                                                                \
      kMSDomain,                                                                                                                                           \
      1,                                                                                                                                                   \
      TIn##_##TOut,                                                                                                                                        \
      kCudaExecutionProvider,                                                                                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("TIn", DataTypeImpl::GetTensorType<TIn>()).TypeConstraint("TOut", DataTypeImpl::GetTensorType<TOut>()), \
      Name<TIn, TOut>);

template <typename TIn, typename TOut>
Status ReduceAllL2<TIn, TOut>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<TIn>::MappedType CudaTIn;
  typedef typename ToCudaType<TOut>::MappedType CudaTOut;

  // Get Input tensor count.
  const auto total_tensor_count = ctx->InputCount();
  // We only have one tensor per group so
  // grouped_tensor_pointers[i] always contains only one element.
  std::vector<std::vector<void*>> grouped_tensor_pointers(total_tensor_count);
  std::vector<int> tensor_sizes(total_tensor_count);

  for (int i = 0; i < total_tensor_count; ++i) {
    const Tensor* input = ctx->Input<Tensor>(i);
    const auto size = input->Shape().Size();
    ORT_ENFORCE(size <= std::numeric_limits<int>::max(), "Number of reduced elements (",
                size, ") exceeds the max allowed value (", std::numeric_limits<int>::max(), ").");
    grouped_tensor_pointers[i] = {const_cast<TIn*>(input->Data<TIn>())};
    tensor_sizes[i] = static_cast<int>(size);
  }

  // Allocate output tensor.
  Tensor* output = ctx->Output(0, {});
  CudaTOut* p_output = reinterpret_cast<CudaTOut*>(output->template MutableData<TOut>());
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(p_output, 0, sizeof(CudaTOut), Stream()));

  const bool deterministic = ctx->GetUseDeterministicCompute();

  if (!deterministic) {
    typedef MultiTensorReduceL2<CudaTIn, CudaTOut> TFunctor;
    TFunctor functor;

    // Check if all values are finite and write true to deviceOutput.
    // Otherwise, false will be written.
    launch_multi_tensor_functor<1, TFunctor>(Stream(),
                                             2048 * 32, tensor_sizes, grouped_tensor_pointers, functor, p_output);

    // *p_output is the squared sum of all elements.
    // Let's take a sqrt to get the actual L2-norm.
    ScalarSqrt(Stream(), p_output, p_output);
  } else {
    // alternate path only for deterministic compute ..
    typedef AccumulationType_t<CudaTOut> CudaTAcc;

    // find reduction buffer size needed by 'reduce_square_sum' for each tensor
    size_t reduction_buffer_size = 0;
    for (int i = 0; i < total_tensor_count; ++i) {
      reduction_buffer_size =
          std::max(reduction_buffer_size, compute_reduction_buffer_size<CudaTAcc>(tensor_sizes[i]));
    }

    // enlarge reduction buffer size for 'reduce_sum' over tensor square norms
    reduction_buffer_size =
        std::max(reduction_buffer_size, compute_reduction_buffer_size<CudaTAcc>(total_tensor_count));

    // create GPU scratch space and zero target for each tensor square norm
    auto reduction_buffer = GetScratchBuffer<void>(reduction_buffer_size);

    // buffer for final output and square norms of each tensor
    auto results_buffer = GetScratchBuffer<CudaTAcc>(1 + total_tensor_count);

    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(results_buffer.get(), 0, sizeof(CudaTAcc) * (1 + total_tensor_count), Stream()));

    CudaTAcc* p_global_sqnorm = results_buffer.get();
    CudaTAcc* p_tensor_sqnorm = p_global_sqnorm + 1;

    // perform reduction l2norm = sqrt[sum(tensor[i][j]**2)] for i,j over all tensor elements
    for (int i = 0; i < total_tensor_count; ++i) {
      CudaTIn* p_tensor_i = reinterpret_cast<CudaTIn*>(grouped_tensor_pointers[i][0]);
      ORT_RETURN_IF_ERROR(reduce_square_sum(
          Stream(), p_tensor_i, p_tensor_sqnorm + i, tensor_sizes[i], reduction_buffer.get(), reduction_buffer_size));
    }
    ORT_RETURN_IF_ERROR(reduce_sum(
        Stream(), p_tensor_sqnorm, p_global_sqnorm, total_tensor_count, reduction_buffer.get(), reduction_buffer_size));
    ScalarSqrt(Stream(), p_global_sqnorm, p_output);
  }

  return Status::OK();
}

REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, float, float)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, MLFloat16, float)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, float, MLFloat16)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, MLFloat16, MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, BFloat16, float)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, float, BFloat16)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, BFloat16, BFloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime
