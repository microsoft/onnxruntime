// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/aten_functions/aten_function_op.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ATenFunctionOp, kMSDomain, 1, kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    onnxruntime::contrib::ATenFunctionOpBase<false>);

ONNX_OPERATOR_KERNEL_EX(
    ATenFunctionOpGrad, kMSDomain, 1, kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    onnxruntime::contrib::ATenFunctionOpBase<true>);

}  // namespace cuda
}  // namespace onnxruntime
