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
    program_ = onnxruntime::opencl::LoadProgram(exec_->GetOpenCLContext(), exec_->GetOpenCLDevice(), src);
  }

  void LoadProgram(const char* src, size_t src_len) {
    program_ = onnxruntime::opencl::LoadProgram(exec_->GetOpenCLContext(), exec_->GetOpenCLDevice(), src, src_len);
  }

  bool LoadKernel(const char* kernel_name) {
    auto kernel = onnxruntime::opencl::LoadKernel(program_, kernel_name);
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
