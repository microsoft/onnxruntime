// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 10, MatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, MatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kMSDomain, 1, FusedMatMul);

}  // namespace opencl
}  // namespace onnxruntime
