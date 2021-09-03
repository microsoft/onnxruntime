// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cpu/tensor/shape_op.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    TransposeOfShape,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        // properly force CPU/GPU synch inside the kernel
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    TransposeOfShape);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
