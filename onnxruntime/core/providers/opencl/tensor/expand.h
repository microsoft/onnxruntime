

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, float, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, double, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int32_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, int64_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, uint32_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, uint64_t, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, bool, Expand);

}  // namespace opencl
}  // namespace onnxruntime
