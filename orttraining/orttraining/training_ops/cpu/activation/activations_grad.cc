// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "orttraining/training_ops/cpu/activation/activations_grad.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    GeluGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    GeluGrad<float>);

ONNX_OPERATOR_KERNEL_EX(
    FastGeluGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FastGeluGrad<float>);

template<typename T>
constexpr T FastGeluGrad<T>::kAlpha;

template<typename T>
constexpr T FastGeluGrad<T>::kBeta;

template<typename T>
constexpr T FastGeluGrad<T>::kGamma;

}  // namespace contrib
}  // namespace onnxruntime
