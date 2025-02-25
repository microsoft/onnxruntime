// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_transfer.h"

#include "opencl_execution_provider.h"
#include "opencl_utils.h"

namespace onnxruntime {
namespace opencl {


class OpenCLGPUDataTransfer : public IDataTransfer {
 public:
  OpenCLGPUDataTransfer(const OpenCLExecutionProvider* exec);
  ~OpenCLGPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

 private:
  const OpenCLExecutionProvider* exec_;
  const OpenCLKernelHolder* kernels_;
};



}  // namespace opencl
}  // namespace onnxruntime
