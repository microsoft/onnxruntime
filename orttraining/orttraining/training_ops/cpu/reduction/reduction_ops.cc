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
Status ReduceSumTraining<T>::Compute(OpKernelContext* ctx) const {
  CommonReduce1Loop<ReduceAggregatorSum<T>>(ctx, axes_, keepdims_, noop_with_empty_axes_);
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
