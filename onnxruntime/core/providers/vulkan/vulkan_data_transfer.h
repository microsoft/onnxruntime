// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//
// TODO: Not currently used as the implementation is creating a compiled model. Required to use static kernels.
//

#include <optional>

#include "kompute/Kompute.hpp"

#include "core/common/common.h"
#include "core/framework/data_transfer.h"
#include "core/providers/vulkan/vulkan_memory_allocator.h"

namespace onnxruntime {
namespace vulkan {

class VulkanDataTransferImpl {
 public:
  VulkanDataTransferImpl(kp::Manager& manager);
  ~VulkanDataTransferImpl() = default;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  common::Status CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const;

 private:
  common::Status CopyTensorImpl(const Tensor& src, Tensor& dst) const;

  kp::Manager& manager_;
};

// wrapper as we need to return a unique_ptr from IExecutionProvider::GetDataTransfer but we need to update the
// underlying instance from VulkanExecutionProvider after session state initialization is done.
// TODO: That was required for NCNN. If not required for Kompute we can remove the wrapper.
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

  // common::Status CopyTensorAsync(const Tensor& *src*, Tensor& *dst*, Stream& *stream*) const override

 private:
  const VulkanDataTransferImpl& impl_;
};

}  // namespace vulkan
}  // namespace onnxruntime
