// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"
#include "core/providers/rocm/rocm_fwd.h"

namespace onnxruntime {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    1,
    kRocmExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("T",  DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace rocm
}  // namespace onnxruntime
