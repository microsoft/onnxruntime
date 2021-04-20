// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/external_functions/external_function_op.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ExternalFunctionOp, kMSDomain, 1, kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    onnxruntime::contrib::ExternalFunctionOpBase<false>);

ONNX_OPERATOR_KERNEL_EX(
    ExternalFunctionOpGrad, kMSDomain, 1, kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes()).ExternalOutputs(),
    onnxruntime::contrib::ExternalFunctionOpBase<true>);

}  // namespace cuda
}  // namespace onnxruntime
