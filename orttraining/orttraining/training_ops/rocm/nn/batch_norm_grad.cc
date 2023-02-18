// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/rocm/nn/batch_norm_grad.h"
#include "core/providers/common.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include "core/providers/rocm/math/unary_elementwise_ops_impl.h"

using namespace std;
namespace onnxruntime {
namespace rocm {

#define REGISTER_GRADIENT_KERNEL_TYPED(T, T1, T2)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                             \
      BatchNormalizationGrad,                                                                \
      kMSDomain,                                                                             \
      1,                                                                                     \
      T##_##T1##_##T2,                                                                       \
      kRocmExecutionProvider,                                                                \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>())    \
                                   .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())  \
                                   .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>()), \
      BatchNormalizationGrad<T, T1, T2>);

template <typename T, typename T1, typename T2>
Status BatchNormalizationGrad<T, T1, T2>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<T>::MappedType HipT;
  typedef typename ToHipType<T1>::MappedType HipT1;
  typedef typename ToHipType<T2>::MappedType HipT2;

  const Tensor* dY = ctx->Input<Tensor>(0);
  const Tensor* X = ctx->Input<Tensor>(1);
  const Tensor* Scale = ctx->Input<Tensor>(2);
  const Tensor* saved_mean = ctx->Input<Tensor>(3);
  // miopenBatchNormalizationBackward() claims to use `savedInvVariance`, but the value
  // is actually equal to the batch inv_std, so we use name `saved_inv_std` here.
  const Tensor* saved_inv_std = ctx->Input<Tensor>(4);
  const TensorShape input_shape = X->Shape();
  const TensorShape channel_shape = saved_mean->Shape();

  // no B here, but B has same size as Scale, so can validate inputs for gradient with this substitute
  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, Scale, Scale, saved_mean, saved_inv_std));

  auto dY_data = reinterpret_cast<const HipT*>(dY->template Data<T>());
  auto X_data = reinterpret_cast<const HipT*>(X->template Data<T>());
  auto Scale_data = reinterpret_cast<const HipT1*>(Scale->template Data<T1>());
  auto saved_mean_data = reinterpret_cast<const HipT2*>(saved_mean->template Data<T2>());
  auto saved_inv_std_data = reinterpret_cast<const HipT2*>(saved_inv_std->template Data<T2>());

  auto dX_data = reinterpret_cast<HipT*>(ctx->Output(0, input_shape)->template MutableData<T>());
  auto dScale_data = reinterpret_cast<HipT1*>(ctx->Output(1, channel_shape)->template MutableData<T1>());
  auto dBias_data = reinterpret_cast<HipT1*>(ctx->Output(2, channel_shape)->template MutableData<T1>());

  const auto alpha = Consts<HipT>::One;
  const auto beta = Consts<HipT>::Zero;

  MiopenTensor input_tensor, scale_bias_tensor;
  vector<int64_t> new_dims;
  BatchNormHelper::NormalizeDims(input_shape, new_dims);
  ORT_RETURN_IF_ERROR(input_tensor.Set(new_dims, MiopenTensor::GetDataType<HipT>()));
  // for fp16 input, `scale_bias_tensor` will have a float type; otherwise it will be the same as input type.
  ORT_RETURN_IF_ERROR(scale_bias_tensor.Set(input_tensor, miopen_batch_norm_mode_));

  const int64_t C = new_dims[1];
  auto p_scale = reinterpret_cast<const void*>(Scale_data);
  auto p_saved_mean = reinterpret_cast<const void*>(saved_mean_data);
  auto p_saved_inv_std = reinterpret_cast<const void*>(saved_inv_std_data);
  auto p_dScale = reinterpret_cast<void*>(dScale_data);
  auto p_dBias = reinterpret_cast<void*>(dBias_data);

  IAllocatorUniquePtr<float> p_f_scale, p_f_dScale, p_f_dBias, p_f_saved_mean, p_f_saved_inv_std;

  if (std::is_same<T1, MLFloat16>::value) {
    p_f_scale = GetScratchBuffer<float>(C, ctx->GetComputeStream());
    p_f_dScale = GetScratchBuffer<float>(C, ctx->GetComputeStream());
    p_f_dBias = GetScratchBuffer<float>(C, ctx->GetComputeStream());

    Impl_Cast<HipT1, float>(Stream(ctx), Scale_data, p_f_scale.get(), C);

    p_scale = p_f_scale.get();
    p_dScale = p_f_dScale.get();
    p_dBias = p_f_dBias.get();
  }

  if (std::is_same<T2, MLFloat16>::value) {
    p_f_saved_mean = GetScratchBuffer<float>(C, ctx->GetComputeStream());
    p_f_saved_inv_std = GetScratchBuffer<float>(C, ctx->GetComputeStream());

    Impl_Cast<HipT2, float>(Stream(ctx), saved_mean_data, p_f_saved_mean.get(), C);
    Impl_Cast<HipT2, float>(Stream(ctx), saved_inv_std_data, p_f_saved_inv_std.get(), C);

    p_saved_mean = p_f_saved_mean.get();
    p_saved_inv_std = p_f_saved_inv_std.get();
  }

  MIOPEN_RETURN_IF_ERROR(miopenBatchNormalizationBackward(
      GetMiopenHandle(ctx),
      miopen_batch_norm_mode_,
      &alpha,
      &beta,
      &alpha,
      &beta,
      input_tensor,
      X_data,
      input_tensor,
      dY_data,
      input_tensor,
      dX_data,
      scale_bias_tensor,
      p_scale,
      p_dScale,
      p_dBias,
      epsilon_,
      p_saved_mean,
      p_saved_inv_std));

  if (std::is_same<T1, MLFloat16>::value) {
    Impl_Cast<float, HipT1>(Stream(ctx), reinterpret_cast<float*>(p_dScale), dScale_data, C);
    Impl_Cast<float, HipT1>(Stream(ctx), reinterpret_cast<float*>(p_dBias), dBias_data, C);
  }

  return Status::OK();
}

#define SPECIALIZED_GRADIENT(T, T1, T2)     \
  REGISTER_GRADIENT_KERNEL_TYPED(T, T1, T2) \
  template Status BatchNormalizationGrad<T, T1, T2>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_GRADIENT(float, float, float)
// MIOpen kernel does not support double, disable for now.
// SPECIALIZED_GRADIENT(double, double, double)
SPECIALIZED_GRADIENT(MLFloat16, MLFloat16, MLFloat16)
SPECIALIZED_GRADIENT(MLFloat16, MLFloat16, float)
SPECIALIZED_GRADIENT(MLFloat16, float, float)

}  // namespace rocm
}  // namespace onnxruntime
