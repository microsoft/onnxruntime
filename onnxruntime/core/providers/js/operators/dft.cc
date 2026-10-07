// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "dft.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DFT,
    kOnnxDomain,
    17, 19,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .InputMemoryType(OrtMemTypeCPU, 1),  // dft_length
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    DFT,
    kOnnxDomain,
    20,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .InputMemoryType(OrtMemTypeCPU, 1)   // dft_length
        .InputMemoryType(OrtMemTypeCPU, 2),  // axis
    DFT);

}  // namespace js
}  // namespace onnxruntime
