// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_data_transfer.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"
#include "core/providers/vulkan/ort_kompute_tensor.h"

namespace onnxruntime {
namespace vulkan {

VulkanDataTransferImpl::VulkanDataTransferImpl(kp::Manager& manager)
    : manager_{manager} {
}

bool VulkanDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) ||
         // not clear if we need to support GPU to GPU to change memory type
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU &&
          src_device.MemType() != dst_device.MemType());
}

Status VulkanDataTransferImpl::CopyTensorImpl(const Tensor& src, Tensor& dst) const {
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_RETURN_IF_NOT(src.ByteOffset() == 0, "Copy with byte offset is not supported");

    // get Tensor from dst data pointer
    // do we vkMapBuffer here or in allocator?
    // copy into staging buffer
    // do we need to set data type/shape etc. in the kp::Tensor? possibly not.

    KomputeTensor* kp_dst = static_cast<KomputeTensor*>(dst.MutableDataRaw());
    kp_dst->SyncWithOrtTensor(dst);  // sync data type and shape

  } else if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_RETURN_IF_NOT(dst.ByteOffset() == 0, "Copy with byte offset is not supported");

    const KomputeTensor* kp_src = static_cast<const KomputeTensor*>(src.DataRaw());
    kp_src->CopyToOrtTensor(dst);

  } else if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_NOT_IMPLEMENTED("Unclear if this is needed");
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "Cannot copy tensor from ", src_device.ToString(),
                           " to ", dst_device.ToString());
  }

  return Status::OK();
}

common::Status VulkanDataTransferImpl::CopyTensor(const Tensor& src, Tensor& dst) const {
  ORT_RETURN_IF_ERROR(CopyTensorImpl(src, dst));
  return Status::OK();
}

common::Status VulkanDataTransferImpl::CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const {
  // we use VkTransfer to copy initializers during session initialization, and VkCompute to copy input/output tensors
  bool copy_to_gpu = src_dst_pairs.front().src.get().Location().device.Type() == OrtDevice::CPU;

  for (const auto& pair : src_dst_pairs) {
    // validate assumption we are only going in one direction
    assert(pair.src.get().Location().device.Type() == (copy_to_gpu ? OrtDevice::CPU : OrtDevice::GPU));
    ORT_RETURN_IF_ERROR(CopyTensorImpl(pair.src, pair.dst));
  }

  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
