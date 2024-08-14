// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "kompute/Kompute.hpp"
#include "core/providers/vulkan/vulkan_memory_allocator.h"

#include "include/ncnn/command.h"
#include "include/ncnn/option.h"

#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/session_options.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/providers/vulkan/vulkan_data_transfer.h"

namespace ncnn {
class Layer;
}

namespace onnxruntime {
class NodeArg;
namespace vulkan {
class VulkanKernel;
}

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
  struct NcnnModel;
  struct KomputeModel;

 public:
  explicit VulkanExecutionProvider(const VulkanExecutionProviderInfo& info);
  virtual ~VulkanExecutionProvider();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanExecutionProvider);

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override {
    return std::make_unique<vulkan::VulkanDataTransfer>(data_transfer_);
  }

  const ncnn::VulkanDevice& Device() const { return ncnn_vulkan_device_; }
  const ncnn::Option& NcnnOptions() const { return ncnn_options_; }

  // temporary hack to enable execution of static kernels without Streams being setup
  kp::Sequence& GetRunSequence() const {}

 private:
  // TODO: If we want to support concurrent execution we need to figure out an efficient way to manage the
  // VkMat instances for constant initializers as the NCNN execution has a std::vector<ncnn::VkMat> for all the values.
  // We'd need to create that on a per-request basis which would mean copying the VkMat info for all the constant
  // initializers.
  bool ConcurrentRunSupported() const override { return false; }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                                const IKernelLookup& kernel_lookup) const override;

  common::Status CompileNcnn(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                             std::vector<NodeComputeInfo>& node_compute_funcs);

  common::Status CompileKompute(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                std::vector<NodeComputeInfo>& node_compute_funcs);

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

  common::Status UploadNcnnConstantInitializers(NcnnModel& model);
  common::Status UploadKomputeConstantInitializers(const GraphViewer& graph_viewer,
                                                   KomputeModel& model);

  common::Status CreateKomputeKernels(KomputeModel& model);

  ncnn::Option ncnn_options_;

  // TODO: ownership is currently in a global in NCNN code. TBD if we want to move it to be local to this instance.
  const ncnn::VulkanDevice& ncnn_vulkan_device_;

  // allocators for various operations.
  // they're in this class so they can be shared between VulkanAllocator and VulkanDataTransfer.
  ncnn::VkWeightStagingAllocator weight_staging_allocator_;
  ncnn::VkWeightAllocator weight_allocator_;

  // Allocators for copying 'blobs' (model input/output) to/from the device. used during inference.
  ncnn::VkStagingAllocator staging_allocator_;
  ncnn::VkBlobAllocator blob_allocator_;

  std::unique_ptr<ncnn::PipelineCache> pipeline_cache_;

  ModelMetadefIdGenerator metadef_id_generator_;

  struct NcnnModel {
    std::vector<std::unique_ptr<vulkan::VulkanKernel>> layers;
    std::vector<size_t> output_indexes;
  };

  struct KomputeModel {
    std::vector<std::unique_ptr<vulkan::VulkanKernel>> layers;
    std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>> constant_initializers;
  };

  // one entry per partition
  std::unordered_map<std::string, std::unique_ptr<NcnnModel>> ncnn_models_;
  std::unordered_map<std::string, std::unique_ptr<KomputeModel>> kompute_models_;

  kp::Manager kompute_manager_;

  bool use_kompute_{true};
  VmaAllocator vma_allocator_{nullptr};  // start with one. might want a separate one for weights

  vulkan::VulkanDataTransferImpl data_transfer_;

  std::shared_ptr<kp::Sequence> run_sequence_;
};

}  // namespace onnxruntime
