// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

// Data transfer manager, which has all functions registered to copy tensors with different location.
// It's not thread-safe.
class IDataTransfer {
 public:
  virtual ~IDataTransfer() = default;

  virtual bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_Device) const = 0;

  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const = 0;
};

class CPUDataTransfer : public IDataTransfer {
 public:
  CPUDataTransfer() = default;
  virtual bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_Device) const override;
  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
};
}  // namespace onnxruntime
