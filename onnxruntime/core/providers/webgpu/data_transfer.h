// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

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
  explicit DataTransferImpl(std::function<const BufferManager&()> buffer_manager_getter)
      : buffer_manager_getter_{std::move(buffer_manager_getter)} {}

  common::Status CopyTensor(void const* src_data,
                            bool src_is_gpu,
                            void* dst_data,
                            bool dst_is_gpu,
                            size_t bytes) const;

 private:
  std::function<const BufferManager&()> buffer_manager_getter_;
};

class DataTransfer : public IDataTransfer {
 public:
  explicit DataTransfer(std::function<const BufferManager&()> buffer_manager_getter)
      : impl_{std::move(buffer_manager_getter)} {}
  ~DataTransfer() {};

  // Device-compatibility half of CanCopy, split out because it needs no BufferManager and so can
  // be tested without a live device.
  static bool IsSupportedDevicePair(const OrtDevice& src_device, const OrtDevice& dst_device);

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

 private:
  DataTransferImpl impl_;
};

}  // namespace webgpu
}  // namespace onnxruntime
