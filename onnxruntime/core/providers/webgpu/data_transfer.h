// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;

class DataTransfer : public IDataTransfer {
 public:
  DataTransfer(const WebGpuContext& context) : context_{context} {};
  ~DataTransfer() {};

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

 private:
  const WebGpuContext& context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
