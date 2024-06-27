// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_data_transfer.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace vulkan {

bool VulkanDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU && src_device.MemType() != dst_device.MemType());
}

Status VulkanDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_RETURN_IF_NOT(src.ByteOffset() == 0, "Copy with byte offset is not supported");
    ORT_NOT_IMPLEMENTED("Copy to device");
  }

  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_RETURN_IF_NOT(dst.ByteOffset() == 0, "Copy with byte offset is not supported");
    ORT_NOT_IMPLEMENTED("Copy from device");
  }

  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_NOT_IMPLEMENTED("Unclear if this is needed");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "Cannot copy tensor from ", src_device.ToString(),
                         " to ", dst_device.ToString());
}

}  // namespace vulkan
}  // namespace onnxruntime
