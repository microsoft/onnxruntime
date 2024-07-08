// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ncnn-src/src/gpu.h"
#include "ncnn-src/src/pipelinecache.h"

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/memcpy.h"
#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_execution_provider.h"
#include "core/providers/vulkan/vulkan_allocator.h"
#include "core/providers/vulkan/vulkan_data_transfer.h"
#include "core/providers/vulkan/vulkan_utils.h"

#include "core/providers/vulkan/activation/activations.h"

namespace onnxruntime {
namespace vulkan {

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

// forward declarations of the kernel classes
class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyToHost, 1);
class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyFromHost, 1);
class ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 6, 12);
class ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 13);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kVulkanExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kVulkanExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

Status RegisterVulkanKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyToHost, 1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(MemcpyFromHost, 1)>,
      BuildKernelCreateInfo<ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 6, 12)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(Sigmoid, 13)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetVulkanKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterVulkanKernels(*kernel_registry));
  return kernel_registry;
}
}  // namespace vulkan

namespace {
ncnn::VulkanDevice& GetVulkanDevice() {
  // get_gpu_count/get_default_gpu_index/get_gpu_device all implicitly create the gpu instance if it doesn't exist yet.
  // there is also `create_gpu_instance(const char* driver_path = 0);` if we need/want

  int gpu_count = ncnn::get_gpu_count();
  LOGS_DEFAULT(INFO) << "Vulkan capable GPU count:" << gpu_count;

  if (gpu_count == 0) {
    ORT_THROW("No Vulkan capable GPU detected.");
  }

  auto device_index = ncnn::get_default_gpu_index();  // TODO: Make device id configurable.

  // TODO: info on the available devices is logged with NCNN_LOGE by create_gpu_instance if NCNN_STDIO is defined.
  // we could maybe look at doing that via ORT logging so the ORT log severity applies.
  auto* device = ncnn::get_gpu_device(device_index);
  ORT_ENFORCE(device, "Failed to get Vulkan device.");

  return *device;
}
}  // namespace

std::unordered_map<std::string, VulkanExecutionProvider::IsSupportedFn>
    VulkanExecutionProvider::node_is_supported_checkers_ = {
        {"Sigmoid", vulkan::Sigmoid::IsSupported},
};

VulkanExecutionProvider::VulkanExecutionProvider(const VulkanExecutionProviderInfo& /*info*/)
    : IExecutionProvider(kVulkanExecutionProvider),
      vulkan_device_{GetVulkanDevice()},
      weight_staging_allocator_{&vulkan_device_},
      weight_allocator_{&vulkan_device_},  // TODO: preferred_block_size is 8 MB. should it be different or configurable?
      staging_allocator_{&vulkan_device_},
      blob_allocator_{&vulkan_device_},  // TODO: preferred_block_size is 16 MB. should it be different or configurable?
      data_transfer_{vulkan_device_, weight_staging_allocator_, weight_allocator_},
      pipeline_cache_{std::make_unique<ncnn::PipelineCache>(&vulkan_device_)} {
  ncnn_options_.use_vulkan_compute = true;

  // allocators used for initializers and transfers between GPU and CPU
  ncnn_options_.pipeline_cache = pipeline_cache_.get();
}

common::Status VulkanExecutionProvider::OnSessionInitializationEnd() {
  data_transfer_.UpdateAllocators(staging_allocator_, blob_allocator_);
  return Status::OK();
}

common::Status VulkanExecutionProvider::OnRunStart(const RunOptions& /*run_options*/) {
  // https://github.com/Tencent/ncnn/wiki/vulkan-notes
  // Do we need to acquire/reclaim on a per-Run basis or could we use staging_allocator_/blob_allocator_?
  // TODO: Figure out how this looks with underlying larger memory allocations to determine what is optional.
  // Assuming we want a small number of large allocations that are bound to buffers. The buffers may come and go
  // more frequently.
  ncnn_options_.blob_vkallocator = vulkan_device_.acquire_blob_allocator();
  ncnn_options_.workspace_vkallocator = ncnn_options_.blob_vkallocator;
  ncnn_options_.staging_vkallocator = vulkan_device_.acquire_staging_allocator();
  return Status::OK();
}

common::Status VulkanExecutionProvider::OnRunEnd(bool /*sync_stream*/, const RunOptions& /*run_options*/) {
  vulkan_device_.reclaim_blob_allocator(ncnn_options_.blob_vkallocator);
  vulkan_device_.reclaim_staging_allocator(ncnn_options_.staging_vkallocator);

  return Status::OK();
}

VulkanExecutionProvider::~VulkanExecutionProvider() {
  // TODO: Is there any explicit release of NCNN owned resources required?
}

std::vector<std::unique_ptr<ComputeCapability>>
VulkanExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (const auto& node : graph_viewer.Nodes()) {
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node);
        kernel_create_info != nullptr) {
      bool is_supported = true;
      // NCNN only supports 4D or less
      for (const auto& def : node.InputDefs()) {
        if (def->Exists()) {
          const auto* shape = def->Shape();
          if (!shape || shape->dim_size() > 4) {
            is_supported = false;
            break;
          }
        }
      }

      if (!is_supported) {
        continue;
      }

      auto entry = node_is_supported_checkers_.find(node.OpType());
      if (entry != node_is_supported_checkers_.end()) {
        is_supported = entry->second(node);
        if (!is_supported) {
          continue;
        }
      }

      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

// common::Status VulkanExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
//                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
//   // see Net::load_model
//   // NCNN execution setup involves looping through all the layers defined in the model to create a pipeline for each.
//
//   // initial thought is we want to create a Net or something similar for each group of nodes.
//   // TBD does that change execution though? if not we could just execute each kernel standalone
//   for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
//     // create a Net for the fused_node_and_graph
//     // loop through the layers in the Net and create a pipeline for each
//   }
//
// }

std::shared_ptr<KernelRegistry> VulkanExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = vulkan::GetVulkanKernelRegistry();
  return kernel_registry;
}

std::vector<AllocatorPtr> VulkanExecutionProvider::CreatePreferredAllocators() {
  // TODO: We may want to utilize VMA to ensure the preferred memory is used as this could be used for either
  // discrete or integrated GPUs.
  // VMA is from AMD
  // https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/choosing_memory_type.html

  std::vector<AllocatorPtr> allocators{
      std::make_unique<vulkan::VulkanBufferAllocator>(
          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, /*device_id*/ 0),
          *ncnn_options_.blob_vkallocator),
      // using 'CUDA_PINNED' for staging memory
      std::make_unique<vulkan::VulkanBufferAllocator>(
          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::CUDA_PINNED, /*device_id*/ 0),
          *ncnn_options_.staging_vkallocator),
  };

  return allocators;
}

}  // namespace onnxruntime
