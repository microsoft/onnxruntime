#pragma once

#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 8, 11, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 12, Concat);

}  // namespace opencl
}  // namespace onnxruntime
