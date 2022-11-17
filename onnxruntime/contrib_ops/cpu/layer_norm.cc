// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// LayerNorm was a contrib op but is now part of the ONNX spec
#include "layer_norm.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
// original LayerNormalization contrib op (incorrectly using onnx domain though)
#define REGISTER_CONTRIB_KERNELS(T)                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(LayerNormalization, kOnnxDomain, 1, 16, T, kCpuExecutionProvider, \
                                          KernelDefBuilder()                                                \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
                                              .TypeConstraint("U", DataTypeImpl::GetTensorType<T>())        \
                                              .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),       \
                                          LayerNorm<false>);                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(SimplifiedLayerNormalization, kOnnxDomain, 1, T, kCpuExecutionProvider,     \
                                KernelDefBuilder()                                                          \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                  \
                                    .TypeConstraint("U", DataTypeImpl::GetTensorType<T>())                  \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),                 \
                                LayerNorm<true>);

REGISTER_CONTRIB_KERNELS(float)
REGISTER_CONTRIB_KERNELS(double)

}  // namespace contrib
}  // namespace onnxruntime
