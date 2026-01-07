// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "if.h"

#include "utils.h"

// Defines a kernel creation function for If opset 13
ONNX_CONTROL_FLOW_OPERATOR_VERSIONED_KERNEL_EX(
    If,
    kOnnxDomain,
    /*start version*/ 13, /*end version*/ 15,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If::CreateKernel)

// Defines a kernel creation function for If opset 16
ONNX_CONTROL_FLOW_OPERATOR_VERSIONED_KERNEL_EX(
    If,
    kOnnxDomain,
    /*start version*/ 16, /*end version*/ 18,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If::CreateKernel)

// Defines a kernel creation function for If opset 19
ONNX_CONTROL_FLOW_OPERATOR_VERSIONED_KERNEL_EX(
    If,
    kOnnxDomain,
    /*start version*/ 19, /*end version*/ 20,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If::CreateKernel)

// Defines a kernel creation function for If opset 21
ONNX_CONTROL_FLOW_OPERATOR_VERSIONED_KERNEL_EX(
    If,
    kOnnxDomain,
    /*start version*/ 21, /*end version*/ 22,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If::CreateKernel)

// Defines a kernel creation function for If opset 23
ONNX_CONTROL_FLOW_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    /*version*/ 23,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If::CreateKernel)

// Defines a kernel creation function for If opset 24
ONNX_CONTROL_FLOW_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    /*version*/ 24,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    If::CreateKernel)

/*static*/
OrtStatus* ORT_API_CALL If::CreateKernel(void* /*kernel_create_func_state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  RETURN_IF_ERROR(Ort::GetEpApi().CreateIfKernel(info, kernel_out));
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}
