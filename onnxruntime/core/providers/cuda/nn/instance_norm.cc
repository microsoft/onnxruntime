// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "instance_norm.h"
#include "instance_norm_impl.h"
#include "core/providers/cpu/nn/instance_norm_helper.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"

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

  auto y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto scale_data = reinterpret_cast<const CudaT*>(scale->template Data<T>());
  auto bias_data = reinterpret_cast<const CudaT*>(bias->template Data<T>());

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

    CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
        CudnnHandle(),
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
    ORT_RETURN_IF_ERROR(data_desc.Set({1, stats_count, image_size, 1}, CudnnTensor::GetDataType<CudaT>()));

    CudnnTensor stats_desc;
    ORT_RETURN_IF_ERROR(stats_desc.Set({1, stats_count, 1, 1}, CudnnTensor::GetDataType<CudaT>()));

    auto mean = GetScratchBuffer<CudaT>(stats_count);
    auto variance = GetScratchBuffer<CudaT>(stats_count);
    auto unused_scale = GetScratchBuffer<CudaT>(stats_count);
    auto unused_bias = GetScratchBuffer<CudaT>(stats_count);

    // first, compute mean and variance per-instance per-channel using cudnnBatchNorm training
    CUDNN_RETURN_IF_ERROR(cudnnBatchNormalizationForwardTraining(
        CudnnHandle(),
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
        Stream(),
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
