// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "core/common/common.h"
#include "core/framework/data_transfer.h"

// #include "ncnn-src/src/command.h"
// #include "ncnn-src/src/option.h"

namespace ncnn {
class VkAllocator;
class VkCompute;
class VkTransfer;
class VulkanDevice;
class Option;
}  // namespace ncnn

namespace onnxruntime {
namespace vulkan {

class VulkanDataTransferImpl {
 public:
  VulkanDataTransferImpl(const ncnn::VulkanDevice& vulkan_device, const ncnn::Option& ncnn_options);
  // ncnn::VkAllocator& staging_allocator, ncnn::VkAllocator& device_allocator);
  ~VulkanDataTransferImpl() = default;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  common::Status CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const;

  void SetSessionInitialized() { session_initialized_ = true; }

 private:
  common::Status CopyTensorImpl(const Tensor& src, Tensor& dst,
                                std::optional<ncnn::VkTransfer>& transfer,
                                std::optional<ncnn::VkCompute>& compute) const;

  const ncnn::VulkanDevice& vulkan_device_;
  const ncnn::Option& ncnn_options_;
  bool session_initialized_{false};
};

// wrapper as we need to return a unique_ptr from IExecutionProvider::GetDataTransfer but we need to update the
// underlying instance from VulkanExecutionProvider after session state initialization is done.
class VulkanDataTransfer : public IDataTransfer {
 public:
  VulkanDataTransfer(const VulkanDataTransferImpl& impl) : impl_{impl} {}
  ~VulkanDataTransfer() = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    return impl_.CopyTensor(src, dst);
  }

  common::Status CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const override {
    return impl_.CopyTensors(src_dst_pairs);
  }

  // common::Status CopyTensorAsync(const Tensor& /*src*/, Tensor& /*dst*/, Stream& /*stream*/) const override

 private:
  const VulkanDataTransferImpl& impl_;
};

}  // namespace vulkan
}  // namespace onnxruntime
