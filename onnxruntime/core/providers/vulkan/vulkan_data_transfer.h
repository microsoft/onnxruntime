// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_execution_provider.h"
#include "vulkan_tensor.h"
#include "vulkan_command_pool.h"
#include "vulkan_pipeline.h"
#include "vulkan_buffer.h"
#include "vulkan_sampler.h"

#include "core/framework/data_transfer.h"

namespace onnxruntime {

class VulkanDeviceToHostDataTransferCore {
 public:
  explicit VulkanDeviceToHostDataTransferCore(const VulkanExecutionProvider& vulkan_execution_provider);

  virtual ~VulkanDeviceToHostDataTransferCore() = default;

  Status Copy(const Tensor& src_tensor, Tensor& dst_tensor);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanDeviceToHostDataTransferCore);

 private:
  const VulkanExecutionProvider& vulkan_execution_provider_;
  std::vector<std::shared_ptr<VulkanDescriptorSet>> desc_sets_;
  std::vector<std::shared_ptr<VulkanBuffer>> offsets_;
  std::shared_ptr<VulkanBuffer> dims_buffer_;
  std::shared_ptr<VulkanBuffer> conduit_buffer_;
  VulkanPipeline* pipeline_ = nullptr;
  const VulkanSampler* sampler_ = nullptr;
};

class VulkanHostToDeviceDataTransferCore {
 public:
  explicit VulkanHostToDeviceDataTransferCore(const VulkanExecutionProvider& vulkan_execution_provider);

  virtual ~VulkanHostToDeviceDataTransferCore() = default;

  Status Copy(const Tensor& src_tensor, Tensor& dst_tensor);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanHostToDeviceDataTransferCore);

 private:
  const VulkanExecutionProvider& vulkan_execution_provider_;
  std::vector<std::shared_ptr<VulkanDescriptorSet>> desc_sets_;
  std::vector<std::shared_ptr<VulkanBuffer>> offsets_;
  std::shared_ptr<VulkanBuffer> dims_buffer_;
  std::shared_ptr<VulkanBuffer> conduit_buffer_;
  VulkanPipeline* pipeline_ = nullptr;
  const VulkanSampler* sampler_ = nullptr;
};

class VulkanDataTransfer : public IDataTransfer {
 public:
  explicit VulkanDataTransfer(const VulkanExecutionProvider& vulkan_exeution_provider);
  ~VulkanDataTransfer() = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

 private:
  const VulkanExecutionProvider& vulkan_execution_provider_;
  std::unique_ptr<VulkanDeviceToHostDataTransferCore> device_to_host_core_;
  std::unique_ptr<VulkanHostToDeviceDataTransferCore> host_to_device_core_;
};
}  // namespace onnxruntime