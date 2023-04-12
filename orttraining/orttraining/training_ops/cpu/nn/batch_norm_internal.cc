// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/batch_norm.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    BatchNormInternal, kMSDomain, 1, kCpuExecutionProvider,
    KernelDefBuilder()
        .Alias(3, 1)
        .Alias(4, 2)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    BatchNorm<float>);

}  // namespace contrib
}  // namespace onnxruntime
