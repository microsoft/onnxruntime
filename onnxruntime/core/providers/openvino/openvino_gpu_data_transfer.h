// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_transfer.h"

namespace onnxruntime {

class OVGPUDataTransfer : public IDataTransfer {
 public:
  OVGPUDataTransfer() ;
  ~OVGPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;
};
}  // namespace Onnxruntime
