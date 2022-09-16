// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_data_transfer.h"

namespace onnxruntime {

VulkanDeviceToHostDataTransferCore::VulkanDeviceToHostDataTransferCore(const VulkanExecutionProvider& vulkan_execution_provider)
    : vulkan_execution_provider_(vulkan_execution_provider) {
  sampler_ = &vulkan_execution_provider_.GetCommonSampler();

  dims_buffer_.reset(new VulkanBuffer(vulkan_execution_provider_.GetMemoryPool(), 8 * sizeof(int), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

  pipeline_ = &vulkan_execution_provider_.GetPipeline("glsl_imageTonchw_comp", types);
}

Status VulkanDeviceToHostDataTransferCore::Copy(const Tensor& src_tensor, Tensor& dst_tensor) {
  // TODO: Merge the common code between VulkanDeviceToHostDataTransferCore::Copy() and
  // VulkanHostToDeviceDataTransferCore::Copy() into a common helper

  ORT_ENFORCE(src_tensor.DataType() == DataTypeImpl::GetType<float>(), "Only copying float tensors is currently supported");
  const VulkanTensor& const_src_vulkan_tensor = *reinterpret_cast<const VulkanTensor*>(src_tensor.DataRaw());
  VulkanTensor& src_vulkan_tensor = const_cast<VulkanTensor&>(const_src_vulkan_tensor);
  ORT_ENFORCE(src_vulkan_tensor.DataType() == dst_tensor.DataType(), "Source and destination tensor types don't match");

  ORT_RETURN_IF_ERROR(vulkan_execution_provider_.Sync());

  auto size = VulkanTensor::GetAlignSize(DataTypeImpl::GetType<float>()) * sizeof(float);

  if (conduit_buffer_->Size() < size) {
    conduit_buffer_.reset(new VulkanBuffer(vulkan_execution_provider_.GetMemoryPool(), size, nullptr,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
  }

  int buffer_size = static_cast<int>(conduit_buffer_->Size());
  VkDeviceSize buffer_offset = 0;

  std::shared_ptr<VulkanCommandBuffer> cmd_buffer_shared_ptr(vulkan_execution_provider_.GetCommandPool().AllocBuffer());
  VkCommandBuffer cmd_buffer = cmd_buffer_shared_ptr->Get();

  auto num_images = src_vulkan_tensor.NumImages();

  cmd_buffer_shared_ptr->Begin(0);

  for (size_t i = 0; i < num_images; ++i) {
    src_vulkan_tensor.GetMutableImage(i)->BarrierRead(cmd_buffer);
  }

  const auto& nhwc = VulkanTensor::TensorShapeFormat(src_tensor.Shape());

  auto* dims = reinterpret_cast<int*>(dims_buffer_->Map());  // W, H, C, N

  dims[0] = static_cast<int>(nhwc[2]);
  dims[1] = static_cast<int>(nhwc[1]);
  dims[2] = static_cast<int>(nhwc[3]);
  dims[3] = static_cast<int>(nhwc[0]);

  dims[4] = 1;
  dims[5] = static_cast<int>(nhwc[2]);
  dims[6] = static_cast<int>(nhwc[2] * nhwc[1]);
  dims[7] = static_cast<int>(nhwc[3] * nhwc[2] * nhwc[1]);

  dims_buffer_->Unmap();

  auto& blocks = src_vulkan_tensor.Blocks();
  auto& limits = vulkan_execution_provider_.GetMemoryLimits();
  int64_t w_unit = limits.maxImageDimension2D;
  int64_t h_unit = limits.maxImageDimension2D;

  struct OffsetBuffer {
    int64_t offset[4];  // Offset w, h, c, n
    int64_t size[4];    //w, h, c, w*h*c
  };

  desc_sets_.resize(num_images);
  offsets_.resize(num_images);

  for (int64_t y = 0; y < blocks[1]; ++y) {
    auto y_start = y * h_unit;

    for (int64_t x = 0; x < blocks[0]; ++x) {
      auto x_start = x * w_unit;

      OffsetBuffer offset;
      offset.offset[0] = x_start;
      offset.offset[1] = y_start;

      auto index = y * blocks[0] + x;
      auto* image = src_vulkan_tensor.GetMutableImage(index);

      offset.size[0] = image->GetWidth();
      offset.size[1] = image->GetHeight();
      offset.size[2] = 0;
      offset.size[3] = image->GetWidth() * image->GetHeight();

      offsets_[index].reset(new VulkanBuffer(vulkan_execution_provider_.GetMemoryPool(), sizeof(offset), &offset, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
      desc_sets_[index].reset(pipeline_->CreateSet());
      desc_sets_[index]->WriteImage(image->GetView(), sampler_->Get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0);
      desc_sets_[index]->WriteBuffer(conduit_buffer_->Get(), 1, buffer_size, buffer_offset);

      desc_sets_[index]->WriteBuffer(dims_buffer_->Get(), 2, dims_buffer_->Size());
      desc_sets_[index]->WriteBuffer(offsets_[index]->Get(), 3, offsets_[index]->Size());
      pipeline_->Bind(cmd_buffer, desc_sets_[index]->Get());

      auto group_count_x = UP_DIV(offset.size[3], 256);
      VK_CALL_RETURNS_VOID(vkCmdDispatch(cmd_buffer, static_cast<uint32_t>(group_count_x), 1, 1));
    }
  }

  ORT_RETURN_IF_ERROR(vulkan_execution_provider_.QueueCommand(cmd_buffer));
  ORT_RETURN_IF_ERROR(vulkan_execution_provider_.Sync());

  // Copy over to the host tensor buffer
  memcpy(dst_tensor.MutableDataRaw(), conduit_buffer_->Map(), size);
  conduit_buffer_->Unmap();

  return Status::OK();
}

VulkanHostToDeviceDataTransferCore::VulkanHostToDeviceDataTransferCore(const VulkanExecutionProvider& vulkan_execution_provider)
    : vulkan_execution_provider_(vulkan_execution_provider) {
  sampler_ = &vulkan_execution_provider_.GetCommonSampler();

  dims_buffer_.reset(new VulkanBuffer(vulkan_execution_provider_.GetMemoryPool(), 8 * sizeof(int), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

  pipeline_ = &vulkan_execution_provider_.GetPipeline("glsl_nchwToimage_comp", types);
}

Status VulkanHostToDeviceDataTransferCore::Copy(const Tensor& src_tensor, Tensor& dst_tensor) {
  ORT_ENFORCE(src_tensor.DataType() == DataTypeImpl::GetType<float>(), "Only copying float tensors is currently supported");
  VulkanTensor& dst_vulkan_tensor = *reinterpret_cast<VulkanTensor*>(dst_tensor.MutableDataRaw());
  ORT_ENFORCE(src_tensor.DataType() == dst_vulkan_tensor.DataType(), "Source and destination tensor types don't match");

  ORT_RETURN_IF_ERROR(vulkan_execution_provider_.Sync());

  auto size = VulkanTensor::GetAlignSize(DataTypeImpl::GetType<float>()) * sizeof(float);

  if (conduit_buffer_->Size() < size) {
    conduit_buffer_.reset(new VulkanBuffer(vulkan_execution_provider_.GetMemoryPool(), size, nullptr,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
  }

  int buffer_size = static_cast<int>(conduit_buffer_->Size());
  VkDeviceSize buffer_offset = 0;

  memcpy(conduit_buffer_->Map(), src_tensor.DataRaw(), size);
  conduit_buffer_->Unmap();

  std::shared_ptr<VulkanCommandBuffer> cmd_buffer_shared_ptr(vulkan_execution_provider_.GetCommandPool().AllocBuffer());
  VkCommandBuffer cmd_buffer = cmd_buffer_shared_ptr->Get();

  auto num_images = dst_vulkan_tensor.NumImages();

  cmd_buffer_shared_ptr->Begin(0);

  for (int i = 0; i < num_images; ++i) {
    dst_vulkan_tensor.GetMutableImage(i)->BarrierWrite(cmd_buffer);
  }

  const auto& nhwc = VulkanTensor::TensorShapeFormat(dst_tensor.Shape());

  auto* dims = reinterpret_cast<int*>(dims_buffer_->Map());  // W, H, C, N

  dims[0] = static_cast<int>(nhwc[2]);
  dims[1] = static_cast<int>(nhwc[1]);
  dims[2] = static_cast<int>(nhwc[3]);
  dims[3] = static_cast<int>(nhwc[0]);

  dims[4] = 1;
  dims[5] = static_cast<int>(nhwc[2]);
  dims[6] = static_cast<int>(nhwc[2] * nhwc[1]);
  dims[7] = static_cast<int>(nhwc[3] * nhwc[2] * nhwc[1]);

  dims_buffer_->Unmap();

  auto& blocks = dst_vulkan_tensor.Blocks();
  auto& limits = vulkan_execution_provider_.GetMemoryLimits();
  int64_t w_unit = limits.maxImageDimension2D;
  int64_t h_unit = limits.maxImageDimension2D;

  struct OffsetBuffer {
    int64_t offset[4];  // Offset w, h, c, n
    int64_t size[4];    //w, h, c, w*h*c
  };

  desc_sets_.resize(num_images);
  offsets_.resize(num_images);

  for (int64_t y = 0; y < blocks[1]; ++y) {
    auto y_start = y * h_unit;

    for (int64_t x = 0; x < blocks[0]; ++x) {
      auto x_start = x * w_unit;

      OffsetBuffer offset;
      offset.offset[0] = x_start;
      offset.offset[1] = y_start;

      auto index = y * blocks[0] + x;
      auto* image = dst_vulkan_tensor.GetMutableImage(index);

      offset.size[0] = image->GetWidth();
      offset.size[1] = image->GetHeight();
      offset.size[2] = 0;
      offset.size[3] = image->GetWidth() * image->GetHeight();

      offsets_[index].reset(new VulkanBuffer(vulkan_execution_provider_.GetMemoryPool(), sizeof(offset), &offset, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
      desc_sets_[index].reset(pipeline_->CreateSet());
      desc_sets_[index]->WriteImage(image->GetView(), sampler_->Get(), VK_IMAGE_LAYOUT_GENERAL, 0);
      desc_sets_[index]->WriteBuffer(conduit_buffer_->Get(), 1, buffer_size, buffer_offset);

      desc_sets_[index]->WriteBuffer(dims_buffer_->Get(), 2, dims_buffer_->Size());
      desc_sets_[index]->WriteBuffer(offsets_[index]->Get(), 3, offsets_[index]->Size());
      pipeline_->Bind(cmd_buffer, desc_sets_[index]->Get());

      auto group_count_x = UP_DIV(offset.size[3], 256);
      VK_CALL_RETURNS_VOID(vkCmdDispatch(cmd_buffer, static_cast<uint32_t>(group_count_x), 1, 1));
    }
  }

  for (int i = 0; i < num_images; ++i) {
    dst_vulkan_tensor.GetMutableImage(i)->BarrierRead(cmd_buffer);
  }

  cmd_buffer_shared_ptr->End();

  ORT_RETURN_IF_ERROR(vulkan_execution_provider_.QueueCommand(cmd_buffer));
  ORT_RETURN_IF_ERROR(vulkan_execution_provider_.Sync());

  return Status::OK();
}

VulkanDataTransfer::VulkanDataTransfer(const VulkanExecutionProvider& vulkan_execution_provider)
    : vulkan_execution_provider_(vulkan_execution_provider) {
  device_to_host_core_ = std::make_unique<VulkanDeviceToHostDataTransferCore>(vulkan_execution_provider_);
  host_to_device_core_ = std::make_unique<VulkanHostToDeviceDataTransferCore>(vulkan_execution_provider_);
}

bool VulkanDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU ||
         dst_device.Type() == OrtDevice::GPU;
}

common::Status VulkanDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {
  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    return host_to_device_core_->Copy(src, dst);
  } else if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    return device_to_host_core_->Copy(src, dst);
  } else {
    // copying between Vulkan memory
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Copying between Vulkan memory is not yet supported");
  }

  // return Status::OK();
}

}  // namespace onnxruntime
