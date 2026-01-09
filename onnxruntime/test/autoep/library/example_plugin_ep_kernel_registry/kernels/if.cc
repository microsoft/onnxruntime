// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "if.h"

#include "utils.h"

// Defines a kernel creation function for If opset 21
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    If,
    kOnnxDomain,
    /*start version*/ 21, /*end version*/ 22,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    IfHelper)

// Defines a kernel creation function for If opset 23
ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    /*version*/ 23,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    IfHelper)

// Defines a kernel creation function for If opset 24
ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    /*version*/ 24,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    IfHelper)

/*static*/
OrtStatus* IfHelper::CreateKernelImpl(const OrtKernelInfo* info, void* /*state*/,
                                      /*out*/ OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  RETURN_IF_ERROR(Ort::GetEpApi().CreateIfKernel(info, &kernel));
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}
