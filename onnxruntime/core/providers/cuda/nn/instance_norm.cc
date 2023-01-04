// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "instance_norm.h"
#include "instance_norm_impl.h"
#include "core/providers/cpu/nn/instance_norm_helper.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      InstanceNormalization,                                      \
      kOnnxDomain,                                                \
      6,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      InstanceNorm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
InstanceNorm<T>::InstanceNorm(const OpKernelInfo& op_kernel_info)
    : CudaKernel(op_kernel_info) {
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = ClampCudnnBatchNormEpsilon(tmp_epsilon);
}

template <typename T>
Status InstanceNorm<T>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(X, scale, bias));

  const TensorShape& x_shape = X->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  auto* y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  const auto* x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  const auto* scale_data = reinterpret_cast<const CudaT*>(scale->Data<T>());
  const auto* bias_data = reinterpret_cast<const CudaT*>(bias->Data<T>());

  const auto& x_dims = x_shape.GetDims();
  const int64_t N = x_dims[0];
  const int64_t C = x_dims[1];
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  if (N == 1) {
    // when N == 1, we can treat it as spatial batch normalization in training
    // as the mean/variance would be computed from input

    CudnnTensor data_desc;
    std::vector<int64_t> new_dims;
    BatchNormHelper::NormalizeDims(x_shape, new_dims);
    ORT_RETURN_IF_ERROR(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));

    CudnnTensor stats_desc;
    ORT_RETURN_IF_ERROR(stats_desc.Set(data_desc, CUDNN_BATCHNORM_SPATIAL));

    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardTrainingHelper(
                              GetCudnnHandle(p_op_kernel_context),
                              CUDNN_BATCHNORM_SPATIAL,
                              &one,
                              &zero,
                              data_desc,
                              x_data,
                              data_desc,
                              y_data,
                              stats_desc,
                              scale_data,
                              bias_data,
                              1.0f,
                              nullptr,
                              nullptr,
                              epsilon_,
                              nullptr,
                              nullptr));
  } else {
    // we use cudnnBatchNormalizationForwardTraining to compute mean/variance
    // so collapsing NC into channel

    auto input_count = x_shape.Size();              // N * C * H * W
    auto stats_count = x_shape.SizeToDimension(2);  // N * C
    auto image_size = input_count / stats_count;

    CudnnTensor data_desc;
    ORT_RETURN_IF_ERROR(data_desc.Set(std::array<int64_t, 4>{1, stats_count, image_size, 1}, CudnnTensor::GetDataType<CudaT>()));

    CudnnTensor stats_desc;
    ORT_RETURN_IF_ERROR(stats_desc.Set(std::array<int64_t, 4>{1, stats_count, 1, 1}, CudnnTensor::GetDataType<CudaT>()));

    const size_t stats_byte_count = stats_count * sizeof(CudaT);

    // Mean & Variance are inputs & outputs and must be initialized to zero to work properly
    auto mean = GetScratchBuffer<CudaT>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mean.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));
    auto variance = GetScratchBuffer<CudaT>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(variance.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));

    // We must set the scale & bias inputs to zero as they are inputs to the calculation
    auto unused_scale = GetScratchBuffer<CudaT>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(unused_scale.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));
    auto unused_bias = GetScratchBuffer<CudaT>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(unused_bias.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));

    // first, compute mean and variance per-instance per-channel using cudnnBatchNorm training
    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardTrainingHelper(
                              GetCudnnHandle(p_op_kernel_context),
                              CUDNN_BATCHNORM_SPATIAL,
                              &one,
                              &zero,
                              data_desc,
                              x_data,
                              data_desc,
                              y_data,  // use y temporarily, would be rewritten later
                              stats_desc,
                              unused_scale.get(),
                              unused_bias.get(),
                              1.0f,
                              mean.get(),
                              variance.get(),
                              CUDNN_BN_MIN_EPSILON,
                              nullptr,
                              nullptr));

    // Y = scale * (x - mean) / sqrt (variance + epsilon) + B
    // X/Y is (N,C,H,W)
    // scale/bias is (1,C,1,1)
    // mean/stddev is (N,C,1,1)
    // NOTE cudnnBatchNormalization computes unbiased variance sum((Xi - mean)^2) / (count - 1)
    // and it needs to be corrected with (count - 1) / count
    fast_divmod fdm_HW(gsl::narrow_cast<int>(image_size));
    fast_divmod fdm_C(gsl::narrow_cast<int>(C));

    InstanceNormImpl<CudaT>(
        Stream(p_op_kernel_context),
        x_data,
        scale_data,
        bias_data,
        mean.get(),
        variance.get(),
        (image_size - 1.0) / image_size,
        static_cast<double>(epsilon_),
        fdm_HW,
        fdm_C,
        y_data,
        input_count);
  }
  return Status::OK();
}

template <>
Status InstanceNorm<MLFloat16>::ComputeInternal(OpKernelContext* p_op_kernel_context) const {
  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;

  const Tensor* X = p_op_kernel_context->Input<Tensor>(0);
  const Tensor* scale = p_op_kernel_context->Input<Tensor>(1);
  const Tensor* bias = p_op_kernel_context->Input<Tensor>(2);

  ORT_RETURN_IF_ERROR(InstanceNormHelper::ValidateInputs(X, scale, bias));

  const TensorShape& x_shape = X->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  auto* y_data = reinterpret_cast<CudaT*>(Y->MutableData<MLFloat16>());
  const auto* x_data = reinterpret_cast<const CudaT*>(X->Data<MLFloat16>());
  const auto* scale_data = reinterpret_cast<const CudaT*>(scale->Data<MLFloat16>());
  const auto* bias_data = reinterpret_cast<const CudaT*>(bias->Data<MLFloat16>());

  const auto& x_dims = x_shape.GetDims();
  const int64_t N = x_dims[0];
  const int64_t C = x_dims[1];
  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  if (N == 1) {
    // when N == 1, we can treat it as spatial batch normalization in training
    // as the mean/variance would be computed from input

    CudnnTensor data_desc;
    std::vector<int64_t> new_dims;
    BatchNormHelper::NormalizeDims(x_shape, new_dims);
    ORT_RETURN_IF_ERROR(data_desc.Set(new_dims, CudnnTensor::GetDataType<CudaT>()));

    CudnnTensor stats_desc;
    ORT_RETURN_IF_ERROR(stats_desc.Set(data_desc, CUDNN_BATCHNORM_SPATIAL));

    // For half input data type, alpha, beta, scale, bias need to be float type.
    // alpha, beta will be of type float as the Consts struct specialization
    // for MLFloat16 type take care of that. Only Convert the scale, bias to float)

    auto scale_data_fp32 = GetScratchBuffer<float>(C, p_op_kernel_context->GetComputeStream());
    Impl_Cast<CudaT, float>(Stream(p_op_kernel_context), scale_data, scale_data_fp32.get(), C);

    auto bias_data_fp32 = GetScratchBuffer<float>(C, p_op_kernel_context->GetComputeStream());
    Impl_Cast<CudaT, float>(Stream(p_op_kernel_context), bias_data, bias_data_fp32.get(), C);

    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardTrainingHelper(
                              GetCudnnHandle(p_op_kernel_context),
                              CUDNN_BATCHNORM_SPATIAL,
                              &one,
                              &zero,
                              data_desc,
                              x_data,
                              data_desc,
                              y_data,
                              stats_desc,
                              scale_data_fp32.get(),
                              bias_data_fp32.get(),
                              1.0f,
                              nullptr,
                              nullptr,
                              epsilon_,
                              nullptr,
                              nullptr));
  } else {
    // we use cudnnBatchNormalizationForwardTraining to compute mean/variance
    // so collapsing NC into channel

    auto input_count = x_shape.Size();              // N * C * H * W
    auto stats_count = x_shape.SizeToDimension(2);  // N * C
    auto image_size = input_count / stats_count;

    CudnnTensor data_desc;
    ORT_RETURN_IF_ERROR(data_desc.Set(std::array<int64_t, 4>{1, stats_count, image_size, 1},
                                      CudnnTensor::GetDataType<CudaT>()));

    // stats_desc needs to be of 'float' type even for float16 input as the "stats" are of float type
    CudnnTensor stats_desc;
    ORT_RETURN_IF_ERROR(stats_desc.Set(std::array<int64_t, 4>{1, stats_count, 1, 1},
                                       CudnnTensor::GetDataType<float>()));

    // For half input data type, we need to allocate some "intermediate"
    // float buffers for CuDNN to use.
    const size_t stats_byte_count = stats_count * sizeof(float);

    // Mean & Variance are inputs & outputs and must be initialized to zero to work properly
    auto mean = GetScratchBuffer<float>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mean.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));
    auto variance = GetScratchBuffer<float>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(variance.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));

    // We must set the scale & bias inputs to zero as they are inputs to the calculation
    auto unused_scale = GetScratchBuffer<float>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(unused_scale.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));
    auto unused_bias = GetScratchBuffer<float>(stats_count, p_op_kernel_context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(unused_bias.get(), 0, stats_byte_count, Stream(p_op_kernel_context)));

    // first, compute mean and variance per-instance per-channel using cudnnBatchNorm training
    CUDNN_RETURN_IF_ERROR(BatchNormalizationForwardTrainingHelper(
                              GetCudnnHandle(p_op_kernel_context),
                              CUDNN_BATCHNORM_SPATIAL,
                              &one,
                              &zero,
                              data_desc,
                              x_data,
                              data_desc,
                              y_data,  // use y temporarily, would be rewritten later
                              stats_desc,
                              unused_scale.get(),
                              unused_bias.get(),
                              1.0f,
                              mean.get(),
                              variance.get(),
                              CUDNN_BN_MIN_EPSILON,
                              nullptr,
                              nullptr));

    // Y = scale * (x - mean) / sqrt (variance + epsilon) + B
    // X/Y is (N,C,H,W)
    // scale/bias is (1,C,1,1)
    // mean/stddev is (N,C,1,1)
    // NOTE cudnnBatchNormalization computes unbiased variance sum((Xi - mean)^2) / (count - 1)
    // and it needs to be corrected with (count - 1) / count
    fast_divmod fdm_HW(gsl::narrow_cast<int>(image_size));
    fast_divmod fdm_C(gsl::narrow_cast<int>(C));

    // The InstanceNormImpl kernel handles the mean/variance in float32, so no casting required here
    InstanceNormImpl<CudaT, float>(
        Stream(p_op_kernel_context),
        x_data,
        scale_data,
        bias_data,
        mean.get(),
        variance.get(),
        (image_size - 1.0) / image_size,
        static_cast<double>(epsilon_),
        fdm_HW,
        fdm_C,
        y_data,
        input_count);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
