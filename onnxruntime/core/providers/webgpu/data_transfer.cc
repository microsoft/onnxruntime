// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"

#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

bool DataTransfer::IsSupportedDevicePair(const OrtDevice& src_device, const OrtDevice& dst_device, int context_id) {
  // WebGPU allocations carry VendorIds::NONE. A vendor-tagged GPU handle belongs to another EP, and
  // reinterpreting it as a WGPUBuffer would be unsafe. The plugin EP transfer applies the same rule.
  if (src_device.Type() == OrtDevice::GPU &&
      (src_device.Vendor() != OrtDevice::VendorIds::NONE || src_device.Id() != context_id)) {
    return false;
  }
  if (dst_device.Type() == OrtDevice::GPU &&
      (dst_device.Vendor() != OrtDevice::VendorIds::NONE || dst_device.Id() != context_id)) {
    return false;
  }

  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return IsSupportedDevicePair(src_device, dst_device, context_id_);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return context_.CopyTensor(buffer_manager_, recording_, src.DataRaw(),
                             src.Location().device.Type() == OrtDevice::GPU,
                             dst.MutableDataRaw(),
                             dst.Location().device.Type() == OrtDevice::GPU,
                             src.SizeInBytes());
}

}  // namespace webgpu
}  // namespace onnxruntime
