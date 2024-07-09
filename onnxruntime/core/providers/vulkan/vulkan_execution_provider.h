// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "ncnn-src/src/command.h"
#include "ncnn-src/src/option.h"

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/providers/vulkan/vulkan_data_transfer.h"

namespace onnxruntime {

namespace {
template <typename T>
T GetProviderOptionWithDefault(const ProviderOptions& options, const std::string& option_name, T default_value) {
  T value = default_value;
  if (auto it = options.find(option_name); it != options.end()) {
    std::istringstream(it->second) >> value;
  }

  return value;
}
}  // namespace

// placeholder for future use. no options currently.
// ProviderOptions are passed through by the generic EP registration infrastructure.
struct VulkanExecutionProviderInfo {
  const SessionOptions* session_options{nullptr};
  VulkanExecutionProviderInfo() = default;

  VulkanExecutionProviderInfo(const ProviderOptions& provider_options, const SessionOptions* sess_option)
      : session_options(sess_option),
        device_id{GetProviderOptionWithDefault<int16_t>(provider_options, "device_id", 0)} {
  }

  const int16_t device_id{0};
};

class VulkanExecutionProvider : public IExecutionProvider {
 public:
  explicit VulkanExecutionProvider(const VulkanExecutionProviderInfo& info);
  virtual ~VulkanExecutionProvider();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanExecutionProvider);

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override {
    return std::make_unique<vulkan::VulkanDataTransfer>(data_transfer_);
  }

  const ncnn::VulkanDevice& Device() const { return vulkan_device_; }
  const ncnn::Option& NcnnOptions() const { return ncnn_options_; }
  // ncnn::VkCompute Compute() const { return *compute_; }

 private:
  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                                const IKernelLookup& kernel_lookup) const override;

  // common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
  //                        std::vector<NodeComputeInfo>& node_compute_funcs) override;

  common::Status OnSessionInitializationEnd() override;
  //common::Status OnRunStart(const RunOptions& /*run_options*/) override {
  //  compute_.emplace(vulkan_device_);
  //}
  //common::Status OnRunEnd(bool /*sync_stream*/, const RunOptions& /*run_options*/) override {
  //  compute_.reset();
  //}

  // in this initial version we allocate/release buffers based on OnRunStart/OnRunEnd using ncnn_options_, and that
  // approach doesn't work with a concurrent run.
  // even if we had an ncnn::Option instance for the allocators for each run it's not clear how we would use those
  // as I'm not sure there's a way to know a call to allocator memory is coming from a specific run.
  bool ConcurrentRunSupported() const override { return false; }

  using IsSupportedFn = std::function<bool(const onnxruntime::Node&)>;
  static std::unordered_map<std::string, IsSupportedFn> node_is_supported_checkers_;

  ncnn::Option ncnn_options_;

  // TODO: ownership is currently a global in NCNN code. TBD if we want to move it to be local to this instance.
  const ncnn::VulkanDevice& vulkan_device_;

  // allocators for various operations.
  // they're in this class so they can be shared between VulkanAllocator and VulkanDataTransfer.
  ncnn::VkWeightStagingAllocator weight_staging_allocator_;
  ncnn::VkWeightAllocator weight_allocator_;

  // Allocators for copying 'blobs' (model input/output) to/from the device. used during inference.
  ncnn::VkStagingAllocator staging_allocator_;
  ncnn::VkBlobAllocator blob_allocator_;

  vulkan::VulkanDataTransferImpl data_transfer_;
  // std::optional<ncnn::VkCompute> compute_;

  std::unique_ptr<ncnn::PipelineCache> pipeline_cache_;
};

}  // namespace onnxruntime
