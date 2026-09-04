// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/providers/cpu/activation/activation_fp16_support.h"
#include "core/providers/cpu/fp16/fp16_activations.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/activations.h"
#endif

using namespace onnxruntime::common;

namespace onnxruntime {

#define REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(op, since_version, end_version) \
  ONNX_CPU_OPERATOR_VERSIONED_KERNEL(                                               \
      op, since_version, end_version,                                               \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), op<float>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(op, since_version) \
  ONNX_CPU_OPERATOR_KERNEL(                                  \
      op, since_version,                                     \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), op<float>);

#define REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(op, since_version, end_version, type) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                                     \
      op, since_version, end_version, type,                                                     \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), op<type>);

#define REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(op, since_version, type) \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                        \
      op, since_version, type,                                           \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), op<type>);

REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(Elu, 6, 21);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 22);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6, 21);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 22);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6, 15);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 6, 12, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 6, 12, double);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 13, 13, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 13, 13, double);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, float);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, double);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, int8_t);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, int32_t);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 6, 12, MLFloat16);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 13, 13, MLFloat16);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, MLFloat16);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(LeakyRelu, 6, 15, MLFloat16);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(LeakyRelu, 16, MLFloat16);
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED

REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(Selu, 6, 21);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 22);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 6, 12, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 6, 12, double);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 13, float);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 13, double);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(Softplus, 1, 21);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softplus, 22);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(Softsign, 1, 21);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 22);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 6, 12, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 6, 12, double);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 6, 12, MLFloat16);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 13, MLFloat16);
#endif
REGISTER_UNARY_ELEMENTWISE_KERNEL(Celu, 12);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 13, float);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 13, double);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 10, 21);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 22);

// Opset-16 adds BFloat16 to allowed types for the LeakyRelu operator
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 16);

namespace functors {
template <typename T>
Status ElementWiseRangedTransform<T>::Create(const std::string& type, const NodeAttributes& attributes,
                                             std::unique_ptr<ElementWiseRangedTransform<T>>& out) {
#define CREATE_ELE_KERNEL(X)                     \
  if (type == #X) {                              \
    auto p = std::make_unique<functors::X<T>>(); \
    ORT_RETURN_IF_ERROR(p->Init(attributes));    \
    out = std::move(p);                          \
    return Status::OK();                         \
  }

  CREATE_ELE_KERNEL(Celu);
  CREATE_ELE_KERNEL(Elu);
  CREATE_ELE_KERNEL(HardSigmoid);
  CREATE_ELE_KERNEL(LeakyRelu);
  CREATE_ELE_KERNEL(Softplus);
  CREATE_ELE_KERNEL(Relu);
  CREATE_ELE_KERNEL(Sigmoid);
  CREATE_ELE_KERNEL(Softsign);
  CREATE_ELE_KERNEL(Tanh);
  CREATE_ELE_KERNEL(ThresholdedRelu);
  CREATE_ELE_KERNEL(Selu);
#ifndef DISABLE_CONTRIB_OPS
  CREATE_ELE_KERNEL(ParametricSoftplus);
  CREATE_ELE_KERNEL(ScaledTanh);
#endif

#undef CREATE_ELE_KERNEL

  return Status(ONNXRUNTIME, FAIL, "unknown kernel type");
}

template Status ElementWiseRangedTransform<float>::Create(const std::string& type, const NodeAttributes& attributes,
                                                          std::unique_ptr<ElementWiseRangedTransform<float>>& out);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
// The MLFloat16 factory only covers the activations that have an MLFloat16
// CPU kernel (see the REGISTER_*_TYPED_KERNEL(..., MLFloat16) list above).
// GemmActivationFusion must not fuse anything outside this set, or FusedGemm
// construction fails at session initialization. IsFp16FusableActivation()
// below is the single source of truth shared by both.
bool IsFp16FusableActivation(const std::string& type) {
  return type == "Relu" || type == "LeakyRelu" || type == "Tanh";
}

template <>
Status ElementWiseRangedTransform<MLFloat16>::Create(
    const std::string& type,
    const NodeAttributes& attributes,
    std::unique_ptr<ElementWiseRangedTransform<MLFloat16>>& out) {
#define CREATE_ELE_KERNEL_FP16(X)                        \
  if (type == #X) {                                      \
    auto p = std::make_unique<functors::X<MLFloat16>>(); \
    ORT_RETURN_IF_ERROR(p->Init(attributes));            \
    out = std::move(p);                                  \
    return Status::OK();                                 \
  }

  CREATE_ELE_KERNEL_FP16(Relu);
  CREATE_ELE_KERNEL_FP16(LeakyRelu);
  CREATE_ELE_KERNEL_FP16(Tanh);

#undef CREATE_ELE_KERNEL_FP16

  return Status(ONNXRUNTIME, NOT_IMPLEMENTED,
                "ElementWiseRangedTransform<MLFloat16> supports only Relu, LeakyRelu and Tanh, got " + type);
}
#endif
}  // namespace functors

namespace functors {
template <>
void Sigmoid<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
  ptrdiff_t len = last - first;
  float* output_ptr = output + first;
  MlasComputeLogistic(input + first, output_ptr, static_cast<size_t>(len));
}

template <>
void Tanh<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
  ptrdiff_t len = last - first;
  float* output_ptr = output + first;
  MlasComputeTanh(input + first, output_ptr, static_cast<size_t>(len));
}
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
template <>
void Tanh<MLFloat16>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
  ptrdiff_t len = last - first;
  MLFloat16* output_ptr = output + first;
  // MlasComputeTanh<T> is templated — MLFloat16 specialization exists
  // in tanh_kernel_neon_fp16.cpp, runs native FP16 NEON on Graviton3
  MlasComputeTanh(input + first, output_ptr, static_cast<size_t>(len));
}
#endif
}  // namespace functors

}  // namespace onnxruntime
