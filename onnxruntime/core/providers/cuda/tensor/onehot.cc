// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/onehot.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

// T1: indices, T2: depth, T3: values
#define REGISTER_TYPED_ONE_HOT_OP(in_type, out_type, depth_type)           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      OneHot,                                                              \
      kOnnxDomain,                                                         \
      11,                                                                  \
      in_type##_##out_type##_##depth_type,                                 \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .InputMemoryType(OrtMemTypeCPUInput, 1) /* Keep depth in CPU */  \
          .InputMemoryType(OrtMemTypeCPUInput, 2) /* Keep values in CPU */ \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<depth_type>()) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<out_type>()),  \
      OneHotOp<in_type, out_type, depth_type>);

REGISTER_TYPED_ONE_HOT_OP(int64_t, int64_t, int64_t)
REGISTER_TYPED_ONE_HOT_OP(int64_t, float, int64_t)
REGISTER_TYPED_ONE_HOT_OP(int32_t, float, int32_t)
REGISTER_TYPED_ONE_HOT_OP(int64_t, MLFloat16, int64_t)
REGISTER_TYPED_ONE_HOT_OP(int32_t, MLFloat16, int32_t)

template <typename in_type, typename out_type, typename depth_type>
Status OneHotOp<in_type, out_type, depth_type>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<out_type>::MappedType CudaT_Out;

  const Tensor* indices = ctx->Input<Tensor>(0);
  const Tensor* depth = ctx->Input<Tensor>(1);
  const Tensor* values = ctx->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(ValidateInputs(depth, values));

  const auto* depth_data = depth->Data<depth_type>();
  const auto depth_val = static_cast<int64_t>(
      *depth_data);  // As per spec in case 'depth' is of non-integer type, it will be casted to int64 before use.
  if (depth_val <= 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Depth is negative.");
  }

  // prepare output shape
  int64_t prefix_dim_size, suffix_dim_size;
  std::vector<int64_t> output_shape;
  ORT_RETURN_IF_ERROR(PrepareOutputShape(indices, depth_val, axis_, prefix_dim_size, suffix_dim_size, output_shape));

  // allocate output
  const auto* values_data = reinterpret_cast<const CudaT_Out*>(values->Data<out_type>());
  Tensor* output = ctx->Output(0, TensorShape(output_shape));

  // edge case where we have a dim with a value of 0
  if (output->Shape().Size() == 0)
    return Status::OK();

  const fast_divmod fdm_suffix(gsl::narrow_cast<int>(suffix_dim_size));
  const auto* indices_data = indices->Data<in_type>();
  auto* output_data = reinterpret_cast<CudaT_Out*>(output->MutableData<out_type>());

  if (values_data[0] == CudaT_Out(0.f)) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output->MutableDataRaw(), 0, output->SizeInBytes(), Stream()));
    OneHotWithZeroOffValueImpl(Stream(),
                               indices_data,
                               fdm_suffix,
                               depth_val,
                               values_data[1],
                               output_data,
                               indices->Shape().Size());
    return Status::OK();
  }

  const fast_divmod fdm_depth_suffix(gsl::narrow_cast<int>(depth_val * suffix_dim_size));
  OneHotImpl(Stream(),
             indices_data, fdm_depth_suffix, fdm_suffix, depth_val,
             values_data[1],
             values_data[0],
             output_data,
             output->Shape().Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
