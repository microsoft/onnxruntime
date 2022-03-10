#pragma once

#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, 12, Resize);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Resize);

}  // namespace opencl
}  // namespace onnxruntime
