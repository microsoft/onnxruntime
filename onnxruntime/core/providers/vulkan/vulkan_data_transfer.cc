// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


/**** NOT CURRENTLY USED *****

Would require cmake/patches/ncnn/record_download.patch.

The whole setup was a little hacky as it tried to replicate internal NCNN packing logic. 


#include "core/providers/vulkan/vulkan_data_transfer.h"

#include "ncnn-src/src/command.h"
#include "ncnn-src/src/gpu.h"
#include "ncnn-src/src/option.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"
#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
namespace vulkan {

VulkanDataTransferImpl::VulkanDataTransferImpl(const ncnn::VulkanDevice& vulkan_device, const ncnn::Option& options)
    : vulkan_device_{vulkan_device},
      ncnn_options_{options} {
}

bool VulkanDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) ||
         // not clear if we need to support GPU to GPU to change memory type
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU &&
          src_device.MemType() != dst_device.MemType());
}

Status VulkanDataTransferImpl::CopyTensorImpl(const Tensor& src, Tensor& dst,
                                              std::optional<ncnn::VkTransfer>& transfer,
                                              std::optional<ncnn::VkCompute>& compute) const {
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  //
  // If copying an initializer we use the VkWeightAllocator/WkWeightStagingAllocator
  // If copying an input/output we use VkBlobAllocator/VkStagingAllocator
  //
  // We should be able to determine the total size of initializers and do one big backing allocation for them,
  // and keep that valid the whole time.
  //
  // If the VkMat is mappable (integrated GPU) we could try and optimize to avoid the allocation/copy.
  // If it requires buffers at the Vulkan level it might not be possible to simply use CPU memory allocated by ORT.
  // There are also nuances. We probably want to use device local + host cached memory for performance which means
  // some flushing logic might be required. Things also differ by device (e.g. AMD vs Intel vs other).
  // See https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them
  // https://www.gdcvault.com/play/1025458/Advanced-Graphics-Techniques-Tutorial-New
  //

  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_RETURN_IF_NOT(src.ByteOffset() == 0, "Copy with byte offset is not supported");

    ncnn::Mat src_mat = TensorToMat(src);
    ncnn::VkMat dst_vkmat = TensorToVkMatWithPacking(dst, *ncnn_options_.blob_vkallocator,
                                                     vulkan_device_, ncnn_options_);

    ORT_ENFORCE(src_mat.total() * src_mat.elemsize == dst_vkmat.total() * dst_vkmat.elemsize,
                "Buffer sizes don't match.");

    const auto* dst_data = dst_vkmat.data;

    if (transfer) {
      // this optionally flattens, but besides that does not change the packing of the data unless it's a 32-bit
      // type and `(opt.use_fp16_storage || (opt.use_fp16_packed && src.elempack % 4 == 0))` is true
      // in which case it casts to fp16
      transfer->record_upload(src_mat, dst_vkmat, ncnn_options_, *flatten* false);
    } else {
      compute->record_upload(src_mat, dst_vkmat, ncnn_options_);
    }

    // if NCNN allocates a new buffer it won't match the address in the tensor that the kernel Compute will receive
    ORT_ENFORCE(dst_data == dst_vkmat.data, "VkMat data changed during copy.");

  } else if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_RETURN_IF_NOT(dst.ByteOffset() == 0, "Copy with byte offset is not supported");

    // TODO: validation assumption that the 'with packaging' info is consistent across the NCNN layers so that it's
    // valid to setup src_vkmat with that logic. It probably doesn't matter unless we actually change data types
    // when going between CPU and GPU (i.e. automatic fp32 <-> fp16 conversion)
    ncnn::VkMat src_vkmat = TensorToVkMatWithPacking(src, *ncnn_options_.blob_vkallocator,
                                                     vulkan_device_, ncnn_options_);
    ncnn::Mat dst_mat = TensorToMat(dst);
    RETURN_IF_NCNN_ERROR(compute->record_download(src_vkmat, dst_mat, ncnn_options_));

  } else if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_NOT_IMPLEMENTED("Unclear if this is needed");
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "Cannot copy tensor from ", src_device.ToString(),
                           " to ", dst_device.ToString());
  }

  return Status::OK();
}

common::Status VulkanDataTransferImpl::CopyTensor(const Tensor& src, Tensor& dst) const {
  // TODO: Figure out what we need to use here to get data to/from gpu in the correct layout.
  //
  // VkTransfer is used by upload_model and has a flag to skip flattening the data. ORT Tensor is not channel aligned
  // so assuming we don't need to flatten.
  // VkCompute is used in NetPrivate::forward_layer and Extractor::extract in net.cpp. It uses barriers which
  // assumably allows more things to happen asynchronously. It also uses Packing_vulkan to pack the data. Not clear
  // if/when we need that to happen.
  //

  std::optional<ncnn::VkTransfer> transfer;
  std::optional<ncnn::VkCompute> compute;

  // we use VkTransfer to copy initializers during session initialization, and VkCompute to copy input/output tensors
  bool use_transfer = !session_initialized_ && src.Location().device.Type() == OrtDevice::CPU;

  if (use_transfer) {
    transfer.emplace(&vulkan_device_);
  } else {
    compute.emplace(&vulkan_device_);
  }

  ORT_RETURN_IF_ERROR(CopyTensorImpl(src, dst, transfer, compute));

  if (use_transfer) {
    RETURN_IF_NCNN_ERROR(transfer->submit_and_wait());
  } else {
    RETURN_IF_NCNN_ERROR(compute->submit_and_wait());
  }

  return Status::OK();
}

common::Status VulkanDataTransferImpl::CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const {
  std::optional<ncnn::VkTransfer> transfer;
  std::optional<ncnn::VkCompute> compute;

  // we use VkTransfer to copy initializers during session initialization, and VkCompute to copy input/output tensors
  bool copy_to_gpu = src_dst_pairs.front().src.get().Location().device.Type() == OrtDevice::CPU;
  bool use_transfer = !session_initialized_ && copy_to_gpu;

  if (use_transfer) {
    transfer.emplace(&vulkan_device_);
  } else {
    compute.emplace(&vulkan_device_);
  }

  for (const auto& pair : src_dst_pairs) {
    // validate assumption we are only going in one direction
    assert(pair.src.get().Location().device.Type() == (copy_to_gpu ? OrtDevice::CPU : OrtDevice::GPU));
    ORT_RETURN_IF_ERROR(CopyTensorImpl(pair.src, pair.dst, transfer, compute));
  }

  if (use_transfer) {
    RETURN_IF_NCNN_ERROR(transfer->submit_and_wait());
  } else {
    RETURN_IF_NCNN_ERROR(compute->submit_and_wait());
  }

  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime

****/