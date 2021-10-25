#pragma once

#include "core/providers/opencl/opencl_onnxruntime_utils.h"

namespace onnxruntime {
namespace opencl {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Add);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Sub);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Mul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Div);

}  // namespace opencl
}  // namespace onnxruntime
