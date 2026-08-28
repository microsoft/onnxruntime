// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "core/common/status.h"
#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;
struct CommandRecordingState;

// Low-level data transfer implementation that operates on raw pointers.
// Used by both DataTransfer (IDataTransfer subclass) and the C API data transfer wrapper.
class DataTransferImpl {
 public:
  DataTransferImpl(const BufferManager& buffer_manager, CommandRecordingState& recording)
      : buffer_manager_{buffer_manager}, recording_{recording} {}

  common::Status CopyTensor(void const* src_data,
                            bool src_is_gpu,
                            void* dst_data,
                            bool dst_is_gpu,
                            size_t bytes) const;

 private:
  mutable std::mutex mutex_;
  const BufferManager& buffer_manager_;
  CommandRecordingState& recording_;
};

class DataTransfer : public IDataTransfer {
 public:
  DataTransfer(const BufferManager& buffer_manager, CommandRecordingState& recording)
      : impl_{buffer_manager, recording} {}
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
