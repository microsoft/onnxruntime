// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "contrib_ops/cpu/activations.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    ParametricSoftplus,
    1,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ParametricSoftplus<float>);

ONNX_CPU_OPERATOR_KERNEL(
    ScaledTanh,
    1,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ScaledTanh<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ThresholdedRelu,
    1,
    9,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ThresholdedRelu<float>);

ONNX_OPERATOR_KERNEL_EX(
    QuickGelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    QuickGelu<float>);

}  // namespace contrib
}  // namespace onnxruntime
