// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "opencl_forward_decl.h"
#include "opencl_execution_provider.h"
#include "opencl_program_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

class OpenCLKernel : public OpKernel, protected OpenCLKernelHolder {
 public:
  explicit OpenCLKernel(const OpKernelInfo& info)
      : OpKernel(info), OpenCLKernelHolder{OPENCL_EXEC_PROVIDER_FROM_INFO(info)->GetProgramManager()}, exec_{OPENCL_EXEC_PROVIDER_FROM_INFO(info)} {
  }

 protected:
  OpenCLExecutionProvider* exec_;
};

}  // namespace opencl
}  // namespace onnxruntime
