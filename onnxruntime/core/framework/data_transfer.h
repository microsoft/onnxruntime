// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

// Data transfer interface.
class IDataTransfer {
 public:
  virtual ~IDataTransfer() = default;

  virtual bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const = 0;

  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const = 0;
  virtual common::Status CopyTensors(const Tensor* src, Tensor* dst, int size) const;
};

class CPUDataTransfer : public IDataTransfer {
 public:
  CPUDataTransfer() = default;
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
};
}  // namespace onnxruntime
