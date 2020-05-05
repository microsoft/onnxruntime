// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/activations.h"
#endif
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

#define CREATE_ELE_KERNEL(X)                  \
  if (type == #X) {                           \
    functors::X<T>* p = new functors::X<T>(); \
    p->Init(attributes);                      \
    out.reset(p);                             \
    return Status::OK();                      \
  }

namespace functors {
template <typename T>
Status ElementWiseRangedTransform<T>::Create(const std::string& type, const NodeAttributes& attributes,
                                             std::unique_ptr<ElementWiseRangedTransform<T>>& out) {
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

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(alias, x, sinceVersion) \
  ONNX_CPU_OPERATOR_KERNEL(                                             \
      alias, sinceVersion,                                              \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), x<float>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion) REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(x, x, sinceVersion)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 10);

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
