#pragma once

#include "opencl_forward_decl.h"
#include "opencl_execution_provider.h"
#include "opencl_kernel_holder.h"
#include "core/framework/op_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class OpenCLKernel : public OpKernel, protected OpenCLKernelHolder {
 public:
  explicit OpenCLKernel(const OpKernelInfo& info)
      : OpKernel(info), exec_(OPENCL_EXEC_PROVIDER_FROM_INFO(info)) {
  }

 protected:
  void LoadProgram(const std::string& src) {
    OpenCLKernelHolder::LoadProgram(exec_, src);
  }

  void LoadProgram(const char* src, size_t src_len) {
    OpenCLKernelHolder::LoadProgram(exec_, src, src_len);
  }

  cl::CommandQueue GetCommandQueue() const {
    return exec_->GetCommandQueue();
  }

 protected:
  OpenCLExecutionProvider* exec_;
};

}  // namespace opencl
}  // namespace onnxruntime
