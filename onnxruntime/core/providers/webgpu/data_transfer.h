// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;

class DataTransfer : public IDataTransfer {
 public:
  DataTransfer(const BufferManager& buffer_manager) : buffer_manager_{buffer_manager} {};
  ~DataTransfer() {};

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

 private:
  const BufferManager& buffer_manager_;
};

}  // namespace webgpu
}  // namespace onnxruntime
