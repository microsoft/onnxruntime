// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_data_transfer.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"
// #include "core/providers/vulkan/vulkan_utils.h"

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

Status VulkanDataTransferImpl::CopyTensorImpl(const Tensor& src, Tensor& dst,
                                              std::optional<kp::Sequence> batch) const {
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_RETURN_IF_NOT(src.ByteOffset() == 0, "Copy with byte offset is not supported");

    // TODO: Map kp::Tensor::TensorDataTypes types to ONNX types
    // Not clear how we'll handle 8-bit data as there's only 1, 4 and 8 bytes data types. not sure if it matters though
    // as the actual type info is plugged in by each kernel.
    // Right now we're only handling float so it doesn't matter
    // TODO: Clarify if the data buffer arg must be non-const. We can't allow modification of the buffer.
    auto kp_tensor = manager_.tensor(const_cast<void*>(src.DataRaw()), src.Shape().Size(), sizeof(float),
                                     kp::Tensor::TensorDataTypes::eFloat);

    if (batch) {
      batch->record<kp::OpTensorSyncDevice>({kp_tensor});
    } else {
      auto seq = manager_.sequence();
      seq->record<kp::OpTensorSyncDevice>({kp_tensor});
      seq->eval();
    }

  } else if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_RETURN_IF_NOT(dst.ByteOffset() == 0, "Copy with byte offset is not supported");

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

**** /
