// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/reduction/reduction_ops.h"
#include "core/providers/common.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/math/unary_elementwise_ops_impl.h"
#include "core/providers/rocm/math/binary_elementwise_ops_impl.h"
#include "core/providers/rocm/math/binary_elementwise_ops.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel_context_internal.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace rocm {

#define REGISTER_MS_KERNEL_TYPED(name, T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      name,                                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, MLFloat16)
REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, float)
REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, double)
// REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, int32_t)

template <bool allow_multi_axes>
template <typename T>
Status ReduceKernel<allow_multi_axes>::ComputeImplEx(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const {
  const Tensor* X = ctx->Input<Tensor>(0);

  //override the attribute value with the input value for reduction_axes
  const Tensor* axes_tensor = ctx->Input<Tensor>(1);
  ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
  ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
  auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
  const auto* data = axes_tensor->template Data<int64_t>();
  std::vector<int64_t> axes(data, data + nDims);

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* Y = ctx->Output(0, X->Shape());
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y->template MutableData<T>(), X->template Data<T>(), X->SizeInBytes(), hipMemcpyDeviceToDevice));
    return Status::OK();
  }

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes,
                                       prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  const bool fast_reduction = fast_reduction_ && !ctx->GetUseDeterministicCompute();

  return ReduceComputeCore<T>(*X, prepare_reduce_metadata, *Y, miopen_reduce_op, axes,
                              calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction);
}

}  // namespace rocm
}  // namespace onnxruntime