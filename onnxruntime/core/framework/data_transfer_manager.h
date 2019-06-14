// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

using DataTransfer = std::function<common::Status(const Tensor&, Tensor& dst, int exec_queue_id)>;

// Data transfer manager, which has all functions registered to copy tensors with different location.
// It's not thread-safe.
class DataTransferManager {
 public:
  static const DataTransferManager& Instance();

  common::Status RegisterDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device, const DataTransfer& data_transfer);

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const;

 private:
  DataTransferManager() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DataTransferManager);

  std::unordered_map<int64_t, DataTransfer> deviceids_datatransfer_map_;
};
}  // namespace onnxruntime
