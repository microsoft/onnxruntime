// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/tensor/gather_grad.h"
#include "core/common/common.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    GatherGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherGrad);

#define TYPED_GRAD_FUNCTION_CALL(T, tp)                                      \
  if (T_type == DataTypeImpl::GetType<T>()) {                                \
    if (Tind_type == DataTypeImpl::GetType<int32_t>()) {                     \
      return ComputeImpl<T, int32_t>(data_shape, indices, grad, output, tp); \
    }                                                                        \
    if (Tind_type == DataTypeImpl::GetType<int64_t>()) {                     \
      return ComputeImpl<T, int64_t>(data_shape, indices, grad, output, tp); \
    }                                                                        \
  }

Status GatherGrad::Compute(OpKernelContext* context) const {
  const Tensor& shape = *context->Input<Tensor>(0);
  const Tensor& indices = *context->Input<Tensor>(1);
  const Tensor& grad = *context->Input<Tensor>(2);

  const TensorShape data_shape(shape.template Data<int64_t>(), shape.Shape().Size());
  Tensor& output = *context->Output(0, data_shape);
  memset(output.MutableDataRaw(), 0, output.SizeInBytes());

  MLDataType T_type = grad.DataType();
  MLDataType Tind_type = indices.DataType();
  TYPED_GRAD_FUNCTION_CALL(float, context->GetOperatorThreadPool());
  TYPED_GRAD_FUNCTION_CALL(double, context->GetOperatorThreadPool());

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for T or Tind not supported yet in GatherGrad.");
}

template <typename T, typename Tind>
Status GatherGrad::ComputeImpl(const TensorShape& data_shape, const Tensor& indices, const Tensor& grad, Tensor& output,
                               concurrency::ThreadPool* tp) const {
  const Tind* indices_data = indices.template Data<Tind>();
  const T* grad_data = grad.template Data<T>();
  T* output_data = output.template MutableData<T>();

  const int64_t axis = HandleNegativeAxis(axis_, data_shape.NumDimensions());
  const int64_t block_size = data_shape.SizeFromDimension(axis + 1);
  const int64_t N = indices.Shape().Size();
  const int64_t input_block_size = data_shape.SizeFromDimension(axis);
  const int64_t output_block_size = N * block_size;
  const int64_t indices_max = data_shape[axis];
  const int64_t grad_size = grad.Shape().Size();

  // Check the indices first in case there's a out of bound index.
  // All index values are expected to be within bounds [-s, s-1] along axis of size s.
  // We can't merge this code in the omp loop below as omp does not allow return in the loop
  for (int64_t i = 0; i < N; i++) {
    Tind idx = indices_data[i];
    if (idx < -indices_max || idx >= indices_max) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices element out of data bounds, idx=", idx,
                             " data_dim=", indices_max);
    }
  }

  std::mutex mtx;
  auto lambda = [&](int64_t g) {
    const int64_t input_block_index = g / output_block_size;
    const int64_t block_offset = g % output_block_size;
    const int64_t indices_index = block_offset / block_size;
    const int64_t offset = block_offset % block_size;
    Tind idx = indices_data[indices_index];
    if (idx < 0) idx += indices_max;
    const int64_t input_index = input_block_index * input_block_size + idx * block_size + offset;
    //REVIEW(codemzs): This lock can become a performance bottleneck. An area for potential improvement.
    std::lock_guard<std::mutex> lck(mtx);
    output_data[input_index] += grad_data[g];
  };

  concurrency::ThreadPool::TryParallelFor(tp, grad_size, static_cast<double>(block_size),
                                          [&lambda](ptrdiff_t first, ptrdiff_t last) {
                                            for (int index = static_cast<int>(first), end = static_cast<int>(last); index < end; ++index) {
                                              lambda(index);
                                            }
                                          });

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
