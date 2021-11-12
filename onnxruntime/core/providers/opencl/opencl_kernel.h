#pragma once

#include "opencl_execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class OpenCLKernel : public OpKernel {
 public:
  explicit OpenCLKernel(const OpKernelInfo& info)
      : OpKernel(info), exec_(OPENCL_EXEC_PROVIDER_FROM_INFO(info)) {
  }

 protected:
  void LoadProgram(const std::string& src) {
    LoadProgram(src.data(), src.size());
  }

  void LoadProgram(const char* src, size_t src_len) {
    cl_int err{};
    program_ = cl::Program(exec_->GetOpenCLContext(), {src, src_len}, /*build=*/true, &err);
    // OPENCL_CHECK_ERROR(err); FIXME: generialize macro
    if (err != CL_SUCCESS) {
      printf("OpenCL Error Code  : %d\n", static_cast<int>(err));
      printf("       Error String: %s\n", onnxruntime::opencl::GetErrorString(err));
      printf("Kernel Source:\n");
      printf("%.*s\n", static_cast<int>(src_len), src);
      auto log = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(exec_->GetOpenCLDevice());
      printf("Build Log:\n");
      printf("%s\n", log.c_str());
      exit(-1);
    }
  }

  bool LoadKernel(const char* kernel_name) {
    cl_int err{};
    cl::Kernel kernel(program_, kernel_name, &err);
    OPENCL_CHECK_ERROR(err);
    kernels_[kernel_name] = kernel;
    return kernels_.insert({kernel_name, kernel}).second;
  }

  cl::Kernel GetKernel(const char* kernel_name) const {
    return kernels_.at(kernel_name);
  }

  cl::CommandQueue GetCommandQueue() const {
    return exec_->GetCommandQueue();
  }

 protected:
  OpenCLExecutionProvider* exec_;

 private:
  cl::Program program_;
  std::unordered_map<std::string, cl::Kernel> kernels_;
};

}  // namespace opencl
}  // namespace onnxruntime
