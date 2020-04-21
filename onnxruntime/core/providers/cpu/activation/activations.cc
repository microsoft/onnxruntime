// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/data_types_internal.h"
#include "core/util/math.h"
#include <cmath>

namespace onnxruntime {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(alias, x, sinceVersion)                              \
  ONNX_CPU_OPERATOR_KERNEL(                                                                          \
      alias,                                                                                         \
      sinceVersion,                                                                                  \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      x<float>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_NONTEMPL(alias, x, sinceVersion, ...)                          \
  ONNX_CPU_OPERATOR_KERNEL(                                                                              \
      alias,                                                                                             \
      sinceVersion,                                                                                      \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<__VA_ARGS__>()), \
      x);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion) \
  REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(x, x, sinceVersion)

REGISTER_UNARY_ELEMENTWISE_KERNEL_NONTEMPL(Elu, Elu, 6, float);
// Not supporting types other than float due to inability to do type inferencing for functions
REGISTER_UNARY_ELEMENTWISE_KERNEL_NONTEMPL(Celu, Celu, 12, float);

REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 6);
// SoftPlus is the default case for ParametricSoftPlus
REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(Softplus, ParametricSoftplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 10);

template <typename T>
struct Elu::EluImpl {
  void operator()(const Tensor*X, Tensor* Y, float alpha) const {
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm >= 0).select(xm, static_cast<T>(alpha) * (xm.exp() - 1));
  }
};

template<>
struct Elu::EluImpl<MLFloat16> {
  void operator()(const Tensor* X, Tensor* Y, float alpha) const {
    Eigen::half alpha_T(alpha);
    Eigen::half zero_T(0.f);
    Eigen::half one_T(1.f);

    ConstEigenVectorArrayMap<Eigen::half> xm(
      reinterpret_cast<const Eigen::half*>(X->template Data<MLFloat16>()), X->Shape().Size());
    EigenVectorArrayMap<Eigen::half>(
      reinterpret_cast<Eigen::half*>(Y->template MutableData<MLFloat16>()), Y->Shape().Size()) =
          (xm >= zero_T).select(xm, alpha_T * (xm.exp() - one_T));
  }
};

Status Elu::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());
  EluImpl<float>()(X, Y, alpha_);
  return Status::OK();
}

namespace celu_internal {
template <typename T>
inline void Compute(const void* input, void* output, float alpha) {
  const auto alpha_T = T{alpha};
  auto input_T = *reinterpret_cast<const T*>(input);
  // max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
  auto expm1_T = std::expm1(input_T / alpha_T);
  auto result = std::max<T>(0, input_T) + std::min<T>(0, alpha_T * expm1_T);
  *reinterpret_cast<T*>(output) = result;
}

inline void Advance(const void*& input, void*& output, size_t element_size) {
  input = reinterpret_cast<const uint8_t*>(input) + element_size;
  output = reinterpret_cast<uint8_t*>(output) + element_size;
}

}  // namespace celu_internal

Status Celu::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());

  const auto element_type = X->GetElementType();
  const auto element_size = X->DataType()->Size();

  const auto* input_data = X->DataRaw();
  const auto* const input_end = reinterpret_cast<const uint8_t*>(input_data) + X->SizeInBytes();
  auto* output_data = Y->MutableDataRaw();

  while (input_data < input_end) {
    switch (element_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        celu_internal::Compute<float>(input_data, output_data, alpha_);
        break;
      default:
        ORT_THROW("Celu: unsupported type: ", X->GetElementType());
    }
    celu_internal::Advance(input_data, output_data, element_size);
  }
  return Status();
}


template <>
Status Sigmoid<float>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  Tensor* Y = context->Output(0, x_shape);
  MlasComputeLogistic(X->template Data<float>(), Y->template MutableData<float>(), x_shape.Size());
  return Status::OK();
}

template <>
Status Tanh<float>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  Tensor* Y = context->Output(0, x_shape);
  MlasComputeTanh(X->template Data<float>(), Y->template MutableData<float>(), x_shape.Size());
  return Status::OK();
}
}  // namespace onnxruntime
