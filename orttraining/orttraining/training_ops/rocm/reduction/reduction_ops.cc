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
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      name<T>);

REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, MLFloat16)
REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, float)
// REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, double)
REGISTER_MS_KERNEL_TYPED(ReduceSumTraining, int32_t)

template <bool allow_multi_axes>
template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices>
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
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y->template MutableData<T>(), X->template Data<T>(), X->SizeInBytes(), hipMemcpyDeviceToDevice, Stream()));
    return Status::OK();
  }

  PrepareReduceMetadata prepare_reduce_metadata;
  ORT_RETURN_IF_ERROR(PrepareForReduce(X,
                                       keepdims_,
                                       axes,
                                       prepare_reduce_metadata));
  Tensor* Y = ctx->Output(0, prepare_reduce_metadata.squeezed_output_dims);
  const bool fast_reduction = fast_reduction_ && !ctx->GetUseDeterministicCompute();

  return ReduceComputeCore<T, ReduceTensorIndices>(*rocm_ep_, *X, prepare_reduce_metadata, *Y, miopen_reduce_op, axes,
                                                   calculate_log_, calculate_sqt_, log_sum_exp_, fast_reduction);
}

template <>
template <>
Status ReduceKernel<true>::ComputeImplEx<int32_t, MIOPEN_REDUCE_TENSOR_NO_INDICES>(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const {
  typedef typename ToHipType<int32_t>::MappedType HipT;

  const Tensor* X = ctx->Input<Tensor>(0);

  //override the attribute value with the input value for reduction_axes
  const Tensor* axes_tensor = ctx->Input<Tensor>(1);
  ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "An axes tensor must be a vector tensor.");
  auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
  const auto* data = axes_tensor->template Data<int64_t>();
  std::vector<int64_t> axes(data, data + nDims);

  // empty axes and no-op
  if (axes.empty() && noop_with_empty_axes_) {
    auto* Y = ctx->Output(0, X->Shape());
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y->template MutableData<int32_t>(), X->template Data<int32_t>(), X->SizeInBytes(), hipMemcpyDeviceToDevice, Stream()));
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
  std::vector<int64_t>& input_dims_miopen = prepare_reduce_metadata.input_dims_miopen;
  std::vector<int64_t>& output_dims_miopen = prepare_reduce_metadata.output_dims_miopen;

  // special case when there is a dim value of 0 in the shape.
  if (input_count == 0) {
    assert(Y->Shape().Size() == 0);
    return Status::OK();
  }

  // miopenReduceTensor for ReduceSum has issue if input and output has same size, we just need to copy the data for this case
  if (input_count == output_count) {
    if (Y->template MutableData<int32_t>() != X->template Data<int32_t>()) {
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(Y->template MutableData<int32_t>(), X->template Data<int32_t>(), input_count * sizeof(int32_t), hipMemcpyDeviceToDevice, Stream()));
    }
    return Status::OK();
  }

  // This reduction keep adding values to this buffer. If a non-zero value, say 1000, is here, the sum will start with 1000.
  // Therefore zeroing out the memory is required
  HIP_RETURN_IF_ERROR(hipMemsetAsync(Y->MutableDataRaw(), 0, Y->SizeInBytes(), Stream()));

  size_t indices_bytes = 0;
  size_t workspace_bytes = 0;
  MiopenTensor input_tensor;
  MiopenTensor output_tensor;
  MiopenReduceDescriptor reduce_desc;

  miopenDataType_t miopen_type_X = miopenFloat;
  IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count);
  Impl_Cast<HipT, float>(Stream(), reinterpret_cast<const HipT*>(X->template Data<int32_t>()), temp_X.get(), X->Shape().Size());

  ORT_RETURN_IF_ERROR(reduce_desc.Set(miopen_reduce_op, miopen_type_X, MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES));
  ORT_RETURN_IF_ERROR(input_tensor.Set(input_dims_miopen, miopen_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(output_dims_miopen, miopen_type_X));
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionIndicesSize(MiopenHandle(), reduce_desc, input_tensor, output_tensor, &indices_bytes));
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionWorkspaceSize(MiopenHandle(), reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  IAllocatorUniquePtr<uint32_t> indices_rocm = GetScratchBuffer<uint32_t>(indices_bytes);
  IAllocatorUniquePtr<HipT> workspace_rocm = GetScratchBuffer<HipT>(workspace_bytes);

  const auto one = Consts<float>::One;
  const auto zero = Consts<float>::Zero;
  auto temp_Y = GetScratchBuffer<float>(output_count);
  MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(MiopenHandle(),
                                            reduce_desc,
                                            indices_rocm.get(),
                                            indices_bytes,
                                            workspace_rocm.get(),
                                            workspace_bytes,
                                            &one,
                                            input_tensor,
                                            temp_X.get(),
                                            &zero,
                                            output_tensor,
                                            temp_Y.get()));

  Impl_Cast<float, int32_t>(Stream(), temp_Y.get(), Y->template MutableData<int32_t>(), output_count);

  return Status::OK();
}

}  // namespace rocm
}  // namespace onnxruntime
