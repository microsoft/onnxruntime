#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class OpenCLExecutionProvider;

namespace opencl {
class OpenCLDataTransfer;
class Image2DDesc;

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

}  // namespace opencl
}  // namespace onnxruntime
