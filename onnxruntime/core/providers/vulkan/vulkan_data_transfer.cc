// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_data_transfer.h"

#include "ncnn-src/src/command.h"
#include "ncnn-src/src/gpu.h"
#include "ncnn-src/src/option.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"
#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
namespace vulkan {

VulkanDataTransferImpl::VulkanDataTransferImpl(const ncnn::VulkanDevice& vulkan_device,
                                               ncnn::VkAllocator& staging_allocator,
                                               ncnn::VkAllocator& device_allocator)
    : vulkan_device_{vulkan_device},
      uploader_{&vulkan_device} {
  ncnn_options_.use_vulkan_compute = true;
  ncnn_options_.staging_vkallocator = &staging_allocator;
  ncnn_options_.blob_vkallocator = &device_allocator;
}

bool VulkanDataTransferImpl::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU &&
          src_device.MemType() != dst_device.MemType());  // DEVICE_LOCAL to HOST_VISIBLE
}

Status VulkanDataTransferImpl::CopyTensor(const Tensor& src, Tensor& dst) const {
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  //
  // If copying an initializer we want to use the VkWeightAllocator/WkWeightStagingAllocator
  // If copying an input/output we want to use VkBlobAllocator/VkStagingAllocator
  //
  // We can probably do better. We should be able to determine the total size of initializers and do
  // one big backing allocation for them, and keep that valid the whole time.
  //

  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_RETURN_IF_NOT(src.ByteOffset() == 0, "Copy with byte offset is not supported");

    // If the VkMat is mappable (integrated GPU) we could try and optimize to avoid the allocation/copy.
    // If it requires buffers at the Vulkan level it might not be possible to simply use CPU memory allocated by ORT.
    // There are also nuances. We probably want to use device local + host cached memory for performance which means
    // some flushing logic might be required. Things also differ by device (e.g. AMD vs Intel vs other).
    // See https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them
    // https://www.gdcvault.com/play/1025458/Advanced-Graphics-Techniques-Tutorial-New
    ncnn::Mat src_mat = TensorToMat(src);
    ncnn::VkMat dst_vkmat = TensorToVkMat(dst, *ncnn_options_.blob_vkallocator);
    if (session_initialized_) {
      // copy of input/output tensors
      ncnn::VkCompute cmd(&vulkan_device_);
      cmd.record_upload(src_mat, dst_vkmat, ncnn_options_);
    } else {
      // copy of read-only initializers
      ncnn::VkTransfer cmd(&vulkan_device_);
      cmd.record_upload(src_mat, dst_vkmat, ncnn_options_);
    }
  }

  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_RETURN_IF_NOT(dst.ByteOffset() == 0, "Copy with byte offset is not supported");

    ncnn::VkCompute cmd(&vulkan_device_);
    ncnn::VkMat src_vkmat = TensorToVkMat(src, *ncnn_options_.blob_vkallocator);
    ncnn::Mat dst_mat = TensorToMat(dst);
    cmd.record_download(src_vkmat, dst_mat, ncnn_options_);
  }

  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_NOT_IMPLEMENTED("Unclear if this is needed");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "Cannot copy tensor from ", src_device.ToString(),
                         " to ", dst_device.ToString());
}

}  // namespace vulkan
}  // namespace onnxruntime
