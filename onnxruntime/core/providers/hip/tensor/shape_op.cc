// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"
#include "core/providers/hip/hip_fwd.h"

namespace onnxruntime {
namespace hip {

ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    1,
    kHipExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("T",  DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace hip
}  // namespace onnxruntime
