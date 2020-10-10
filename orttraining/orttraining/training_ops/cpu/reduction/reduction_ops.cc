// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/reduction/reduction_ops.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/containers.h"
#include "core/platform/threadpool.h"

using namespace std;
namespace onnxruntime {
namespace contrib {

#define REGISTER_REDUCESUMTRAINING_KERNEL_TYPED(T)                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ReduceSumTraining,                                          \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ReduceSumTraining<T>);

REGISTER_REDUCESUMTRAINING_KERNEL_TYPED(float)
REGISTER_REDUCESUMTRAINING_KERNEL_TYPED(double)
REGISTER_REDUCESUMTRAINING_KERNEL_TYPED(int32_t)
REGISTER_REDUCESUMTRAINING_KERNEL_TYPED(int64_t)


template <typename T>
void ReduceSumCore(const T* input_data, T* output_data, bool no_transpose,
                   int64_t blocks, int64_t block_size, FastAllocVector<T>& transposed_input_data,
                   concurrency::ThreadPool* tp) {
  if (no_transpose) {
    auto lambda = [input_data, blocks, output_data](ptrdiff_t i) {
      // The ConstEigenMatrixMap type is expanded to work around a MS compiler issue
      output_data[i] = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(input_data + (i * blocks), blocks).sum();
    };
    concurrency::ThreadPool::TryBatchParallelFor(tp, block_size, lambda, 0);
  } else {
    EigenVectorMap<T> out_vec(output_data, block_size);
    out_vec = ConstEigenMatrixMap<T>(&transposed_input_data[0], block_size, blocks).rowwise().sum();
  }
}

template <typename T>
Status ReduceSumTraining<T>::Compute(OpKernelContext* ctx) const {
  FastAllocVector<T> transposed_input_data(GetAllocator<T>(*ctx));
  int64_t block_size;
  int64_t blocks;
  std::vector<int64_t> reduced_dims;
  const Tensor* input = ctx->Input<Tensor>(0);

  //override the attribute value with the input value for reduction_axes
  const Tensor* axes_tensor = ctx->Input<Tensor>(1);
  ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
  ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
              "An axes tensor must be a vector tensor.");
  auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
  const auto* data = axes_tensor->template Data<int64_t>();
  std::vector<int64_t> axes(data, data + nDims);

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* output = ctx->Output(0, input->Shape());
    memcpy(output->template MutableData<T>(), input->template Data<T>(), input->SizeInBytes());
    return Status::OK();
  }

  bool no_transpose = PrepareForReduce<T>(input, transposed_input_data, block_size, blocks, axes, keepdims_, reduced_dims, true);

  auto* output = ctx->Output(0, reduced_dims);

  ReduceSumCore(input->template Data<T>(), output->template MutableData<T>(),
                no_transpose, blocks, block_size, transposed_input_data, ctx->GetOperatorThreadPool());

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
