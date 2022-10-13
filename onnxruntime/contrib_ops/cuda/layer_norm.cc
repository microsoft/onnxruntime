// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/nn/layer_norm.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// LayerNormalization is an official ONNX operator in opset 17.
#define REGISTER_KERNEL_TYPED(T, U, V)                                                                               \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(LayerNormalization, kOnnxDomain, 1, 16, T##_##U##_##V,                     \
                                          kCudaExecutionProvider,                                                    \
                                          (*KernelDefBuilder::Create())                                              \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                 \
                                              .TypeConstraint("U", DataTypeImpl::GetTensorType<U>())                 \
                                              .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),                \
                                          onnxruntime::cuda::LayerNorm<T, U, V, false>);                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(SimplifiedLayerNormalization, kOnnxDomain, 1, T##_##U##_##V, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                                        \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                           \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<U>())                           \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),                          \
                                onnxruntime::cuda::LayerNorm<T, U, V, true>);

REGISTER_KERNEL_TYPED(float, float, float)
REGISTER_KERNEL_TYPED(double, double, double)
REGISTER_KERNEL_TYPED(MLFloat16, float, MLFloat16)
REGISTER_KERNEL_TYPED(float, float, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, float, float)
REGISTER_KERNEL_TYPED(BFloat16, float, BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
