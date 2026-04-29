// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <span>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/providers/webgpu/buffer_manager.h"

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
#include "core/providers/webgpu/webgpu_pix_frame_generator.h"
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

struct pthreadpool;
namespace onnxruntime {
namespace webgpu {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

class WebGpuContext;
class WebGpuProfiler;
class GpuBufferAllocator;

// Forward declare CapturedCommandInfo which is now defined in webgpu_context.h
struct CapturedCommandInfo;

// The actual implementation of kernel registration.
std::shared_ptr<KernelRegistry> GetKernelRegistry(bool enable_graph_capture, bool enable_int64);
}  // namespace webgpu

struct WebGpuExecutionProviderConfig {
  DataLayout data_layout{DataLayout::NHWC};      // preferred layout is NHWC by default
  bool enable_graph_capture{false};              // graph capture feature is disabled by default
  bool enable_pix_capture{false};                // PIX capture is disabled by default
  bool enable_int64{false};                      // int64 ops are not enabled by default
  uint32_t multi_rotary_cache_concat_offset{0};  // offset for concatenated multi rotary cache (0 = disabled)
  std::vector<std::string> force_cpu_node_names{};
};

class WebGpuExecutionProvider : public IExecutionProvider {
 public:
  WebGpuExecutionProvider(int context_id, webgpu::WebGpuContext& context, WebGpuExecutionProviderConfig&& config);
  ~WebGpuExecutionProvider() override;

  inline auto GetKernelRegistryImpl() const {
    return webgpu::GetKernelRegistry(enable_graph_capture_, enable_int64_);
  }

#if !defined(ORT_USE_EP_API_ADAPTERS)
  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& /*kernel_lookup*/,
      const GraphOptimizerRegistry& /* graph_optimizer_registry */,
      IResourceAccountant* /* resource_accountant */) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return GetKernelRegistryImpl();
  }
#endif
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
#if defined(__wasm__)
  std::unique_ptr<onnxruntime::IExternalDataLoader> GetExternalDataLoader() const override;
#endif

  DataLayout GetPreferredLayout() const override { return preferred_data_layout_; }

  std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                   std::string_view node_op_type,
                                                   DataLayout target_data_layout) const override;

  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  // WebGPU EP disallow concurrent run because actual implementation (eg. WebGPU backend) relies on global states to
  // work, and concurrent run with async function may mess up the states and cause undefined behavior.
  bool ConcurrentRunSupported() const override { return false; }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;
  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  // WebGPU EP reuses the Device ID as the key to get the WebGpuContext instance.
  int GetDeviceId() const override { return context_id_; }

  std::unique_ptr<profiling::EpProfiler> GetProfiler() override;

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured(int graph_annotation_id) const override;
  Status ReplayGraph(int graph_annotation_id) override;
  Status ReleaseGraph(int graph_annotation_id) override;
  OrtGraphCaptureNodeAssignmentPolicy GetGraphCaptureNodeAssignmentPolicy() const override {
    return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
  }
  webgpu::BufferManager& BufferManager() const;
  AllocatorPtr PrepackAllocator() const { return prepack_allocator_; }
  // Set the device allocator pointer so we can call SetBufferManager on it during OnRunStart/OnRunEnd
  void SetDeviceAllocator(webgpu::GpuBufferAllocator* allocator) { default_gpu_allocator_ = allocator; }
  std::span<const std::string> GetForceCpuNodeNames() const { return force_cpu_node_names_; }
  uint32_t MultiRotaryCacheConcatOffset() const { return multi_rotary_cache_concat_offset_; }

#if defined(ORT_USE_EP_API_ADAPTERS)
  inline onnxruntime::ep::adapter::Logger& GetEpLogger() const {
    return *ep_logger_;
  }
  inline void SetEpLogger(const OrtLogger* logger) {
    ep_logger_ = std::make_unique<onnxruntime::ep::adapter::Logger>(logger);
  }
#endif

 private:
  bool IsGraphCaptureAllowed() const;
  void IncrementRegularRunCountBeforeGraphCapture();

  int context_id_;
  webgpu::WebGpuContext& context_;
  webgpu::WebGpuProfiler* session_profiler_{nullptr};
  DataLayout preferred_data_layout_;
  std::vector<std::string> force_cpu_node_names_;
  bool enable_graph_capture_ = false;
  bool enable_int64_ = false;
  uint32_t multi_rotary_cache_concat_offset_ = 0;
  std::unordered_map<int, int> graph_id_to_run_count_;
  const int min_num_runs_before_cuda_graph_capture_ = 1;  // Required regular runs before graph capture for any necessary allocations.
  int m_current_graph_annotation_id = 0;

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  std::unique_ptr<WebGpuPIXFrameGenerator> pix_frame_generator_ = nullptr;
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

  // Default buffer manager for graph capture mode (used during warmup runs
  // and as the stable reference target for GpuBufferAllocator)
  std::unique_ptr<webgpu::BufferManager> graph_default_buffer_mgr_ = nullptr;

  // Per-graph buffer managers keyed by annotation ID.
  // Each captured graph gets its own buffer manager so that buffer caches
  // are isolated between different generators.
  std::unordered_map<int, std::unique_ptr<webgpu::BufferManager>> per_graph_buffer_mgrs_;

  // Store captured commands per graph annotation ID
  std::unordered_map<int, std::vector<webgpu::CapturedCommandInfo>> captured_graphs_;
  // Track which graph annotation IDs have completed capture
  std::unordered_set<int> captured_graph_ids_;

  // Allocator for prepacked weights (uses buffers without mapping)
  AllocatorPtr prepack_allocator_;

  // Raw pointer to the default GPU allocator (owned by the framework via CreatePreferredAllocators)
  // Used to swap the buffer manager for per-graph isolation
  webgpu::GpuBufferAllocator* default_gpu_allocator_ = nullptr;

#if defined(ORT_USE_EP_API_ADAPTERS)
  std::unique_ptr<onnxruntime::ep::adapter::Logger> ep_logger_;
#endif
};

}  // namespace onnxruntime
