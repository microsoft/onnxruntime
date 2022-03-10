#include "opencl_forward_decl.h"
#include "opencl_utils.h"
#include "opencl_execution_provider.h"

namespace onnxruntime {
namespace opencl {

struct OpenCLKernelHolder {
  cl_program program;
  std::unordered_map<std::string, cl_kernel> kernels;

  inline void LoadProgram(const OpenCLExecutionProvider* exec, const std::string& src) {
    program = onnxruntime::opencl::LoadProgram(exec->GetOpenCLContext(), exec->GetOpenCLDevice(), src, exec->UseFp16());
  }

  inline void LoadProgram(const OpenCLExecutionProvider* exec, const char* src, size_t src_len) {
    program = onnxruntime::opencl::LoadProgram(exec->GetOpenCLContext(), exec->GetOpenCLDevice(), src, src_len, exec->UseFp16());
  }

  inline bool LoadKernel(const char* kernel_name) {
    auto kernel = onnxruntime::opencl::LoadKernel(program, kernel_name);
    kernels[kernel_name] = kernel;
    return kernels.insert({kernel_name, std::move(kernel)}).second;
  }

  inline cl_kernel GetKernel(const std::string& kernel_name) const {
    auto it = kernels.find(kernel_name);
    if (it != kernels.end()) {
      return it->second;
    }
    ORT_THROW("Unable to find kernel ", kernel_name);
  }
};

}  // namespace opencl
}  // namespace onnxruntime
