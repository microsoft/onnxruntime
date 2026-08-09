// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul.h"

namespace onnxruntime {
namespace contrib {

// TransposedMatMul is kept for backward compatibility
ONNX_OPERATOR_KERNEL_EX(
    TransposeMatMul,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedMatMul,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedMatMul,
    kMSDomain,
    1,
    double,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

}  // namespace contrib
}  // namespace onnxruntime
