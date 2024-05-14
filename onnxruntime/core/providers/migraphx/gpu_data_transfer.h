// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "migraphx_inc.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer() = default;
  ~GPUDataTransfer() = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;
};

}  // namespace onnxruntime
