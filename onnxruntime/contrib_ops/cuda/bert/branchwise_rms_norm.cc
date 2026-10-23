// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/branchwise_rms_norm.h"

#include <limits>

#include "contrib_ops/cuda/bert/branchwise_rms_norm_impl.h"
#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/providers/cuda/reduction/reduction_functions.h"

namespace onnxruntime::contrib::cuda {

using namespace onnxruntime::cuda;

#define REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(V, T)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      BranchwiseRMSNorm, kMSDomain, V, T, kCudaExecutionProvider,          \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("F", DataTypeImpl::GetTensorType<float>())       \
          .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())    \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                         \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      BranchwiseRMSNorm<T>);

REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(1, float)
REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(1, MLFloat16)
REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(1, BFloat16)
REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(2, float)
REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(2, MLFloat16)
REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL(2, BFloat16)

#undef REGISTER_BRANCHWISE_RMS_NORM_CUDA_KERNEL

template <typename T>
BranchwiseRMSNorm<T>::BranchwiseRMSNorm(const OpKernelInfo& info)
    : CudaKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      use_onnx_arithmetic_(info.GetAttrOrDefault<int64_t>("use_onnx_arithmetic", 0) != 0),
      input_start_(info.GetAttrOrDefault<int64_t>("input_start", 0)),
      input_width_(info.GetAttrOrDefault<int64_t>("input_width", 0)) {
  const auto mode = info.GetAttrOrDefault<int64_t>("use_onnx_arithmetic", 0);
  ORT_ENFORCE(mode == 0 || mode == 1, "use_onnx_arithmetic must be 0 or 1");
  ORT_ENFORCE(input_start_ >= 0 && input_width_ >= 0, "input_start and input_width must be nonnegative");
  ORT_ENFORCE(input_width_ != 0 || input_start_ == 0, "input_start requires input_width");
  ORT_ENFORCE(input_width_ == 0 || (use_onnx_arithmetic_ && num_branches_ == 0),
              "input_width requires use_onnx_arithmetic=1 and grouped input");
}

template <typename T>
Status BranchwiseRMSNorm<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  const auto* numerical_mode = context->Input<Tensor>(2);
  bool dense_prefill = false;
  if (numerical_mode != nullptr) {
    ORT_RETURN_IF_NOT(numerical_mode->Shape().Size() == 1,
                      "prefill_numerics must contain one int32 value");
    const int32_t mode = *numerical_mode->Data<int32_t>();
    ORT_RETURN_IF_NOT(mode == 0 || mode == 1, "prefill_numerics must be 0 or 1");
    dense_prefill = mode == 1;
    ORT_RETURN_IF_NOT(!dense_prefill || use_onnx_arithmetic_,
                      "Dense prefill numerical mode requires use_onnx_arithmetic=1");
  }
  auto output_dims = x->Shape().AsShapeVector();
  ORT_RETURN_IF_NOT(!output_dims.empty(), "BranchwiseRMSNorm requires a non-scalar input");
  const int64_t input_stride = output_dims.back();
  if (input_width_ != 0) {
    ORT_RETURN_IF_NOT(input_start_ <= input_stride && input_width_ <= input_stride - input_start_,
                      "input slice exceeds the last dimension");
    output_dims.back() = input_width_;
  }
  const TensorShape output_shape(output_dims);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(output_shape, num_branches_, params));
  if (scale != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
  }
  ORT_RETURN_IF_NOT(params.branches <= std::numeric_limits<int>::max() &&
                        params.hidden <= std::numeric_limits<int>::max() &&
                        (input_width_ == 0 || input_stride <= std::numeric_limits<int>::max()),
                    "branch or hidden dimension is too large for CUDA");

  const int64_t groups = output_shape.Size() / params.hidden;
  auto* y = context->Output(0, output_shape);
  auto* widened = context->OutputCount() > 1 ? context->Output(1, output_shape) : nullptr;
  ORT_RETURN_IF_NOT(widened == nullptr || use_onnx_arithmetic_,
                    "Y_fp32 requires use_onnx_arithmetic=1");
  if (groups == 0) {
    return Status::OK();
  }

  const auto* x_data = reinterpret_cast<const CudaT*>(x->Data<T>());
  auto* y_data = reinterpret_cast<CudaT*>(y->MutableData<T>());
  if (use_onnx_arithmetic_) {
    ORT_RETURN_IF_NOT(groups <= std::numeric_limits<int>::max(),
                      "ONNX arithmetic requires the row count to fit in int32");
    const auto buffer_size = compute_reduce_matrix_columns_buffer_size<float>(
        static_cast<int>(groups), static_cast<int>(params.hidden));
    auto squares = GetScratchBuffer<float>(groups == 1 ? output_shape.Size() : 0, GetComputeStream(context));
    auto sums = GetScratchBuffer<float>(groups, GetComputeStream(context));
    auto buffer = GetScratchBuffer<uint8_t>(buffer_size, GetComputeStream(context));
    return LaunchStandardBranchwiseRMSNorm<CudaT>(
        Stream(context), x_data + input_start_, scale == nullptr ? nullptr : scale->DataRaw(),
        scale == nullptr ? x->GetElementType() : scale->GetElementType(), y_data,
        static_cast<int>(groups), static_cast<int>(params.branches), static_cast<int>(params.hidden),
        scale == nullptr || scale->Shape().Size() == params.hidden, epsilon_,
        widened == nullptr ? nullptr : widened->MutableData<float>(),
        squares.get(), sums.get(), buffer.get(), buffer_size,
        static_cast<int>(input_width_ == 0 ? params.hidden : input_stride), dense_prefill);
  }
  return LaunchMixedScaleBranchwiseRMSNorm<CudaT>(
      Stream(context), x_data, scale == nullptr ? nullptr : scale->DataRaw(),
      scale == nullptr ? x->GetElementType() : scale->GetElementType(), y_data,
      groups, static_cast<int>(params.branches), static_cast<int>(params.hidden),
      scale == nullptr || scale->Shape().Size() == params.hidden, epsilon_);
}

template class BranchwiseRMSNorm<float>;
template class BranchwiseRMSNorm<MLFloat16>;
template class BranchwiseRMSNorm<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
