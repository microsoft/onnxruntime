// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/reduction/reduction_all.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_REDUCEALLL2_KERNEL_TYPED(TIn, TOut)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(ReduceAllL2, kMSDomain, 1, TIn##_##TOut, kCpuExecutionProvider,   \
                                KernelDefBuilder()                                                \
                                    .TypeConstraint("TIn", DataTypeImpl::GetTensorType<TIn>())    \
                                    .TypeConstraint("TOut", DataTypeImpl::GetTensorType<TOut>()), \
                                ReduceAllL2<TIn, TOut>);

REGISTER_REDUCEALLL2_KERNEL_TYPED(float, float)

template <typename TIn, typename TOut>
Status ReduceAllL2<TIn, TOut>::Compute(OpKernelContext* ctx) const {
  // Get Input tensor count.
  const auto total_tensor_count = ctx->InputCount();
  std::vector<const TIn*> tensor_pointers(total_tensor_count);
  std::vector<int64_t> tensor_sizes(total_tensor_count);

  for (int i = 0; i < total_tensor_count; ++i) {
    const Tensor* input = ctx->Input<Tensor>(i);
    const auto size = input->Shape().Size();
    ORT_ENFORCE(size <= std::numeric_limits<int>::max(), "Number of reduced elements (", size,
                ") exceeds the max allowed value (", std::numeric_limits<int>::max(), ").");
    tensor_pointers[i] = input->template Data<TIn>();
    tensor_sizes[i] = size;
  }

  // Allocate output tensor.
  Tensor* output = ctx->Output(0, {});
  TOut* output_data = output->template MutableData<TOut>();
  *output_data = TOut(0.f);
  // perform reduction l2norm = sqrt[sum(tensor[i][j]**2)] for i,j over all tensor elements
  for (int i = 0; i < total_tensor_count; ++i) {
    *output_data +=
        ReduceAggregatorSumSquare<TIn, TOut>(tensor_sizes[i], tensor_pointers[i][0]).aggall(tensor_pointers[i]);
  }

  *output_data = reduce_sqrt<TOut>(*output_data);
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
