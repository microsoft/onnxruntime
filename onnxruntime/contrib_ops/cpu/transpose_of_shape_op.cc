// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {
namespace contrib {


ONNX_OPERATOR_KERNEL_EX(
    TransposeOfShape,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    TransposeOfShape);

}  // namespace contrib
}  // namespace onnxruntime
