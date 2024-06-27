// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {
namespace vulkan {

class VulkanDataTransfer : public IDataTransfer {
 public:
  VulkanDataTransfer() = default;
  ~VulkanDataTransfer() = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  // common::Status CopyTensorAsync(const Tensor& /*src*/, Tensor& /*dst*/, Stream& /*stream*/) const

 private:
};

}  // namespace vulkan
}  // namespace onnxruntime
