#include "opencl_execution_provider.h"

#include "core/framework/op_kernel.h"

#define ONNX_OPENCL_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kOpenCLExecutionProvider, builder, __VA_ARGS__)

#define OPENCL_EXEC_PROVIDER_FROM_INFO(info) \
  const_cast<OpenCLExecutionProvider*>(static_cast<const OpenCLExecutionProvider*>((info).GetExecutionProvider()))

#define CL_BUFFER_FROM_TENSOR(TENSOR) (*const_cast<cl::Buffer*>(static_cast<const cl::Buffer*>((TENSOR).DataRaw())))


namespace onnxruntime {
namespace opencl {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

}
}  // namespace onnxruntime
