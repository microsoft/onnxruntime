// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/providers/providers.h"
#include "core/providers/vulkan/vulkan_data_transfer.h"
#include "core/providers/vulkan/vulkan_utils.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// placeholder for future use. no options currently.
// ProviderOptions are passed through by the generic EP registration infrastructure.
struct VulkanExecutionProviderInfo {
  const SessionOptions* session_options{nullptr};
  VulkanExecutionProviderInfo() = default;

  VulkanExecutionProviderInfo(const ProviderOptions& /*provider_options*/, const SessionOptions* sess_option)
      : session_options(sess_option) {
    // Can search for relevant session options here
    // if (auto it = provider_options.find("intra_op_num_threads"); it != provider_options.end()) {
    //   xnn_thread_pool_size = std::stoi(it->second);
    // }
  }
};

class VulkanExecutionProvider : public IExecutionProvider {
 public:
  explicit VulkanExecutionProvider(const VulkanExecutionProviderInfo& info);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanExecutionProvider);
  virtual ~VulkanExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override {
    return std::make_unique<vulkan::VulkanDataTransfer>();
  }

 private:
};

}  // namespace onnxruntime
