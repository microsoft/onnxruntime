#pragma once

namespace onnxruntime {

class OpenCLExecutionProvider;

namespace opencl {
class OpenCLDataTransfer;

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

}  // namespace opencl
}  // namespace onnxruntime
