// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_transfer.h"

#include "ncnn-src/src/command.h"
#include "ncnn-src/src/option.h"

namespace ncnn {
class VkAllocator;
class VkTransfer;
class VulkanDevice;
}  // namespace ncnn

namespace onnxruntime {
namespace vulkan {

class VulkanDataTransferImpl {
 public:
  VulkanDataTransferImpl(const ncnn::VulkanDevice& vulkan_device,
                         ncnn::VkAllocator& staging_allocator, ncnn::VkAllocator& device_allocator);
  ~VulkanDataTransferImpl() = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const;

  // after session state initialization is done we switch from VkWeightStagingAllocator/VkWeightAllocator to
  // VkStagingAllocator/VkBlobAllocator
  // NOTE: We currently only support usage of the Vulkan EP in a single InferenceSession so we can use the approach
  //       of switching allocators based on IExecutionProvider::OnSessionInitializationEnd being called.
  void UpdateAllocators(ncnn::VkAllocator& staging_allocator, ncnn::VkAllocator& device_allocator) {
    ncnn_options_.staging_vkallocator = &staging_allocator;
    ncnn_options_.blob_vkallocator = &device_allocator;
    session_initialized_ = true;  // copy of initializers is complete.
  }

 private:
  const ncnn::VulkanDevice& vulkan_device_;
  ncnn::Option ncnn_options_;
  ncnn::VkTransfer uploader_;
  bool session_initialized_{false};
};

// wrapper as we need to return a unique_ptr from IExecutionProvider::GetDataTransfer but we need to update the
// underlying instance from VulkanExecutionProvider after session state initialization is done.
class VulkanDataTransfer : public IDataTransfer {
 public:
  VulkanDataTransfer(const VulkanDataTransferImpl& impl) : impl_{impl} {}
  ~VulkanDataTransfer() = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override {
    return impl_.CanCopy(src_device, dst_device);
  }

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    return impl_.CopyTensor(src, dst);
  }

  // common::Status CopyTensorAsync(const Tensor& /*src*/, Tensor& /*dst*/, Stream& /*stream*/) const override

 private:
  VulkanDataTransferImpl impl_;
};

}  // namespace vulkan
}  // namespace onnxruntime
