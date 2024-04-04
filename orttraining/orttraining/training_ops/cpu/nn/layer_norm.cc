// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/layer_norm.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

// LayerNormGrad

#define REGISTER_KERNEL_TYPED(T)                                                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(LayerNormalizationGrad, kMSDomain, 1, T, kCpuExecutionProvider,           \
                                KernelDefBuilder()                                                        \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<T>())                \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),               \
                                LayerNormGrad<T, false>);                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(SimplifiedLayerNormalizationGrad, kMSDomain, 1, T, kCpuExecutionProvider, \
                                KernelDefBuilder()                                                        \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<T>())                \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),               \
                                LayerNormGrad<T, true>);                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(InvertibleLayerNormalizationGrad, kMSDomain, 1, T, kCpuExecutionProvider, \
                                KernelDefBuilder()                                                        \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<T>())                \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),               \
                                InvertibleLayerNormGrad<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

#undef REGISTER_KERNEL_TYPED

template <typename T, bool simplified>
LayerNormGrad<T, simplified>::LayerNormGrad(const OpKernelInfo& op_kernel_info)
    : OpKernel{op_kernel_info} {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
}

template <typename T, bool simplified>
Status LayerNormGrad<T, simplified>::Compute(OpKernelContext* op_kernel_context) const {
  int input_index = 0;
  const Tensor* Y_grad = op_kernel_context->Input<Tensor>(input_index++);
  const Tensor* X = op_kernel_context->Input<Tensor>(input_index++);
  const auto& X_shape = X->Shape();
  const auto axis = HandleNegativeAxis(axis_, X_shape.NumDimensions());
  ORT_ENFORCE(X_shape.SizeToDimension(gsl::narrow_cast<size_t>(axis)) <= std::numeric_limits<Eigen::Index>::max());
  ORT_ENFORCE(X_shape.SizeFromDimension(gsl::narrow_cast<size_t>(axis)) <= std::numeric_limits<Eigen::Index>::max());
  const auto N = static_cast<Eigen::Index>(X_shape.SizeToDimension(gsl::narrow_cast<size_t>(axis)));
  const auto M = static_cast<Eigen::Index>(X_shape.SizeFromDimension(gsl::narrow_cast<size_t>(axis)));
  ORT_ENFORCE(M != 1);

  const Tensor* scale = op_kernel_context->Input<Tensor>(input_index++);
  const Tensor* mean;
  if (!simplified) {
    mean = op_kernel_context->Input<Tensor>(input_index++);
  }
  const Tensor* inv_std_var = op_kernel_context->Input<Tensor>(input_index);

  const auto& scale_shape = scale->Shape();

  Tensor* X_grad = op_kernel_context->Output(0, X_shape);
  Tensor* scale_grad = op_kernel_context->Output(1, scale_shape);
  Tensor* bias_grad = (!simplified) ? op_kernel_context->Output(2, scale_shape) : nullptr;

  // Note: Eigen has column-major storage order by default
  ConstEigenArrayMap<T> Y_grad_arr{Y_grad->Data<T>(), M, N};
  ConstEigenArrayMap<T> X_arr{X->Data<T>(), M, N};
  ConstEigenVectorArrayMap<T> scale_vec{scale->Data<T>(), M};
  ConstEigenVectorArrayMap<float> mean_vec{simplified ? nullptr : mean->Data<float>(), N};
  ConstEigenVectorArrayMap<float> inv_std_var_vec{inv_std_var->Data<float>(), N};

  EigenArrayMap<T> X_grad_arr{X_grad->MutableData<T>(), M, N};
  EigenVectorArrayMap<T> scale_grad_vec{scale_grad->MutableData<T>(), M};
  EigenVectorArrayMap<T> bias_grad_vec = (!simplified) ? EigenVectorArrayMap<T>{bias_grad->MutableData<T>(), M} : EigenVectorArrayMap<T>{nullptr, 0};

  using Array = Eigen::ArrayXX<T>;
  using RowVector = Eigen::Array<T, 1, Eigen::Dynamic>;

  // A, B, C are calculated as below:
  // A = Y_grad * (X - mean(X)) * inv_std_var
  // B = Y_grad * scale * inv_std_var
  // C = Y_grad * scale * inv_std_var * (X - mean(X)) * inv_std_var

  // Simplified Layer Norm
  // A = Y_grad * X * inv_std_var
  // B = Y_grad * scale * inv_std_var
  // C = Y_grad * scale * inv_std_var * X * inv_std_var
  // A, B, and C are M x N
  Array X_mean_difference_over_std_var;
  if (simplified) {
    X_mean_difference_over_std_var =
        X_arr.rowwise() * inv_std_var_vec.cast<T>().transpose();
  } else {
    X_mean_difference_over_std_var =
        (X_arr.rowwise() - mean_vec.cast<T>().transpose()).rowwise() * inv_std_var_vec.cast<T>().transpose();
  }
  Array A = Y_grad_arr * X_mean_difference_over_std_var;
  Array B = (Y_grad_arr.colwise() * scale_vec).rowwise() * inv_std_var_vec.cast<T>().transpose();
  Array C = B * X_mean_difference_over_std_var;

  RowVector mean_C = C.colwise().mean();  // 1 x N

  if (simplified) {
    X_grad_arr = B - X_mean_difference_over_std_var.rowwise() * mean_C;
  } else {
    RowVector mean_B = B.colwise().mean();  // 1 x N
    X_grad_arr = B.rowwise() - mean_B - X_mean_difference_over_std_var.rowwise() * mean_C;
  }

  if (!simplified) {
    bias_grad_vec = Y_grad_arr.rowwise().sum();
  }

  scale_grad_vec = A.rowwise().sum();

  return Status::OK();
}

template <typename T>
InvertibleLayerNormGrad<T>::InvertibleLayerNormGrad(const OpKernelInfo& op_kernel_info)
    : OpKernel{op_kernel_info} {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
}

template <typename T>
Status InvertibleLayerNormGrad<T>::Compute(OpKernelContext* op_kernel_context) const {
  const Tensor* Y_grad = op_kernel_context->Input<Tensor>(0);
  const Tensor* Y = op_kernel_context->Input<Tensor>(1);
  const Tensor* scale = op_kernel_context->Input<Tensor>(2);
  const Tensor* bias = op_kernel_context->Input<Tensor>(3);
  const Tensor* inv_std_var = op_kernel_context->Input<Tensor>(4);

  const auto& Y_shape = Y_grad->Shape();
  const auto& X_shape = Y_shape;
  const auto axis = HandleNegativeAxis(axis_, X_shape.NumDimensions());
  ORT_ENFORCE(X_shape.SizeToDimension(gsl::narrow_cast<size_t>(axis)) <= std::numeric_limits<Eigen::Index>::max());
  ORT_ENFORCE(X_shape.SizeFromDimension(gsl::narrow_cast<size_t>(axis)) <= std::numeric_limits<Eigen::Index>::max());
  const auto N = static_cast<Eigen::Index>(X_shape.SizeToDimension(gsl::narrow_cast<size_t>(axis)));
  const auto M = static_cast<Eigen::Index>(X_shape.SizeFromDimension(gsl::narrow_cast<size_t>(axis)));
  ORT_ENFORCE(M != 1);
  const auto& scale_shape = scale->Shape();

  Tensor* X_grad = op_kernel_context->Output(0, X_shape);
  Tensor* scale_grad = op_kernel_context->Output(1, scale_shape);
  Tensor* bias_grad = op_kernel_context->Output(2, scale_shape);

  // Note: Eigen has column-major storage order by default
  ConstEigenArrayMap<T> Y_grad_arr{Y_grad->Data<T>(), M, N};
  ConstEigenArrayMap<T> Y_arr{Y->Data<T>(), M, N};
  ConstEigenVectorArrayMap<T> scale_vec{scale->Data<T>(), M};
  ConstEigenVectorArrayMap<T> bias_vec{bias->Data<T>(), M};
  ConstEigenVectorArrayMap<float> inv_std_var_vec{inv_std_var->Data<float>(), N};

  EigenArrayMap<T> X_grad_arr{X_grad->MutableData<T>(), M, N};
  EigenVectorArrayMap<T> scale_grad_vec{scale_grad->MutableData<T>(), M};
  EigenVectorArrayMap<T> bias_grad_vec{bias_grad->MutableData<T>(), M};

  using Array = Eigen::ArrayXX<T>;
  using RowVector = Eigen::Array<T, 1, Eigen::Dynamic>;

  // A, B, C are calculated as below:
  // A = Y_grad * (X - mean(X)) * inv_std_var
  // B = Y_grad * scale * inv_std_var
  // C = Y_grad * scale * inv_std_var * (X - mean(X)) * inv_std_var

  // A, B, and C are M x N
  Array X_mean_difference_over_std_var = (Y_arr.colwise() - bias_vec).colwise() / scale_vec;
  Array A = Y_grad_arr * X_mean_difference_over_std_var;
  Array B = (Y_grad_arr.colwise() * scale_vec).rowwise() * inv_std_var_vec.cast<T>().transpose();
  Array C = B * X_mean_difference_over_std_var;

  // mean_B = mean(Y_grad * scale * inv_std_var)
  RowVector mean_B = B.colwise().mean();  // 1 x N

  // mean_C = mean(Y_grad * scale * inv_std_var * (X - mean(X)) * inv_std_var)
  RowVector mean_C = C.colwise().mean();  // 1 x N

  // X_grad = Y_grad * scale * inv_std_var - mean_B - (X - mean(X)) * inv_std_var * mean_C
  //        = B - mean_B - (X - mean(X)) * inv_std_var * mean_c
  X_grad_arr = B.rowwise() - mean_B - X_mean_difference_over_std_var.rowwise() * mean_C;

  // bias_grad = sum(Y_grad)
  bias_grad_vec = Y_grad_arr.rowwise().sum();

  // scale_grad = sum(Y_grad * (X - mean(X)) * inv_std_var)
  //            = sum(A)
  scale_grad_vec = A.rowwise().sum();

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
