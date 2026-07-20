// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;

// Low-level data transfer implementation that operates on raw pointers.
// Used by both DataTransfer (IDataTransfer subclass) and the C API data transfer wrapper.
class DataTransferImpl {
 public:
  DataTransferImpl(const BufferManager& buffer_manager) : buffer_manager_{buffer_manager} {};

  common::Status CopyTensor(void const* src_data,
                            bool src_is_gpu,
                            void* dst_data,
                            bool dst_is_gpu,
                            size_t bytes) const;

 private:
  const BufferManager& buffer_manager_;
};

class DataTransfer : public IDataTransfer {
 public:
  DataTransfer(const BufferManager& buffer_manager) : impl_{buffer_manager} {};
  ~DataTransfer() {};

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

 private:
  DataTransferImpl impl_;
};

}  // namespace webgpu
}  // namespace onnxruntime
