// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/activations.h"
#endif
#include "core/mlas/inc/mlas.h"

using namespace onnxruntime::common;

namespace onnxruntime {

#define CREATE_ELE_KERNEL(X)                  \
  if (type == #X) {                           \
    functors::X<T>* p = new functors::X<T>(); \
    p->Init(attributes);                      \
    out.reset(p);                             \
    return Status::OK();                      \
  }

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

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 6, 12, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 6, 12, double);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 13, 13, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 13, 13, double);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, float);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, double);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, int8_t);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Relu, 14, int32_t);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 6, 12, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 6, 12, double);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 13, float);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Sigmoid, 13, double);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 6, 12, float);
REGISTER_VERSIONED_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 6, 12, double);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Celu, 12);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 13, float);
REGISTER_UNARY_ELEMENTWISE_TYPED_KERNEL(Tanh, 13, double);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 10);

namespace functors {
template <typename T>
Status ElementWiseRangedTransform<T>::Create(const std::string& type, const NodeAttributes& attributes,
                                             std::unique_ptr<ElementWiseRangedTransform<T>>& out) {
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
  return Status(ONNXRUNTIME, FAIL, "unknown kernel type");
}

template Status ElementWiseRangedTransform<float>::Create(const std::string& type, const NodeAttributes& attributes,
                                                          std::unique_ptr<ElementWiseRangedTransform<float>>& out);
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
}  // namespace functors

}  // namespace onnxruntime
