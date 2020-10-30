// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/reduction/reduction_all.h"
#include "core/providers/rocm/reduction/reduction_functions.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
struct AccumulateType {};
template <>
struct AccumulateType<float> { using type = float; };
template <>
struct AccumulateType<half> { using type = float; };
template <>
struct AccumulateType<double> { using type = double; };
template <typename T>
using AccType = typename AccumulateType<T>::type;

#define REGISTER_REDUCE_ALL_KERNEL_TYPED(Name, TIn, TOut)                                                                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                                                                \
      Name,                                                                                                                                     \
      kMSDomain,                                                                                                                                \
      1,                                                                                                                                        \
      TIn##_##TOut,                                                                                                                             \
      kRocmExecutionProvider,                                                                                                                   \
      KernelDefBuilder().TypeConstraint("TIn", DataTypeImpl::GetTensorType<TIn>()).TypeConstraint("TOut", DataTypeImpl::GetTensorType<TOut>()), \
      Name<TIn, TOut>);

template <typename TIn, typename TOut>
Status ReduceAllL2<TIn, TOut>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<TIn>::MappedType HipTIn;
  typedef typename ToHipType<TOut>::MappedType HipTOut;

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
  HipTOut* p_output = reinterpret_cast<HipTOut*>(output->template MutableData<TOut>());
  ORT_ENFORCE(hipMemset(p_output, 0, sizeof(HipTOut)) == hipSuccess);

  // auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  // bool deterministic = ctx_internal && ctx_internal->GetUseDeterministicCompute();
  bool deterministic = true;
  if (!deterministic) {

    typedef MultiTensorReduceL2<HipTIn, HipTOut> TFunctor;
    TFunctor functor;

    // Check if all values are finite and write true to deviceOutput.
    // Otherwise, false will be written.
    launch_multi_tensor_functor<1, TFunctor, HipTOut*>(
        2048 * 32, tensor_sizes, grouped_tensor_pointers, functor, p_output);

    // *p_output is the squared sum of all elements.
    // Let's take a sqrt to get the actual L2-norm.
    ScalarSqrt(p_output, p_output);
  }
  else {

    // alternate path only for deterministic compute ..
    typedef AccType<HipTOut> HipTAcc;

    // find scratch buffer size needed by 'reduce_square_sum' for each tensor
    int scratch_size = 0;
    for (int i = 0; i < total_tensor_count; ++i) {
      scratch_size = std::max(scratch_size, compute_reduction_buffer_size(sizeof(HipTAcc), tensor_sizes[i]));
    }

    // enlarge scratch buffer size for 'reduce_sum' over tensor square norms
    scratch_size = std::max(scratch_size, compute_reduction_buffer_size(sizeof(HipTAcc), total_tensor_count));

    // add head room for final output and square norms of each tensor
    scratch_size += (1 + total_tensor_count)*sizeof(HipTAcc);

    // create GPU scratch space and zero target for each tensor square norm
    uint8_t* p_scratch = GetScratchBuffer<uint8_t>(scratch_size).get();
    ORT_ENFORCE(hipMemset(p_scratch, 0, sizeof(HipTAcc)*(1 + total_tensor_count)) == hipSuccess);

    HipTAcc* p_global_sqnorm = reinterpret_cast<HipTAcc*>(p_scratch);
    HipTAcc* p_tensor_sqnorm = p_global_sqnorm + 1;
    HipTAcc* p_reduce_buffer = p_tensor_sqnorm + total_tensor_count;
 
    // perform reduction l2norm = sqrt[sum(tensor[i][j]**2)] for i,j over all tensor elements
    for (int i = 0; i < total_tensor_count; ++i) {
      HipTIn* p_tensor_i = reinterpret_cast<HipTIn*>(grouped_tensor_pointers[i][0]);
      reduce_square_sum(p_tensor_i, p_tensor_sqnorm + i, tensor_sizes[i], p_reduce_buffer);
    }
    reduce_sum(p_tensor_sqnorm, p_global_sqnorm, total_tensor_count, p_reduce_buffer);
    ScalarSqrt(p_global_sqnorm, p_output);
  }

  return Status::OK();
}

REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, float, float)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, MLFloat16, float)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, float, MLFloat16)
REGISTER_REDUCE_ALL_KERNEL_TYPED(ReduceAllL2, MLFloat16, MLFloat16)

}  // namespace rocm
}  // namespace onnxruntime