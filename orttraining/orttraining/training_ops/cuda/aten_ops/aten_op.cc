// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ATenOp, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
    onnxruntime::contrib::ATenOp);

ONNX_OPERATOR_KERNEL_EX(ATenOpGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
                            .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
                        onnxruntime::contrib::ATenOpGrad);

}  // namespace cuda
}  // namespace onnxruntime
