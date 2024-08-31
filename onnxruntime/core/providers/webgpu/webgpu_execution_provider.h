// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"

struct pthreadpool;
namespace onnxruntime {
namespace webgpu {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

class WebGpuContext;
enum class BufferCacheMode;
}  // namespace webgpu

struct WebGpuExecutionProviderInfo {
  WebGpuExecutionProviderInfo(DataLayout data_layout1, bool enable_graph_capture1)
      : data_layout{data_layout1},
        enable_graph_capture{enable_graph_capture1},
        storage_buffer_cache_mode{},
        uniform_buffer_cache_mode{},
        query_resolve_buffer_cache_mode{},
        default_buffer_cache_mode{} {}
  DataLayout data_layout;
  bool enable_graph_capture;
  webgpu::BufferCacheMode storage_buffer_cache_mode;
  webgpu::BufferCacheMode uniform_buffer_cache_mode;
  webgpu::BufferCacheMode query_resolve_buffer_cache_mode;
  webgpu::BufferCacheMode default_buffer_cache_mode;
};

class WebGpuExecutionProvider : public IExecutionProvider {
 public:
  WebGpuExecutionProvider(int context_id, const webgpu::WebGpuContext& context, const WebGpuExecutionProviderInfo& info);
  ~WebGpuExecutionProvider() override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& /*kernel_lookup*/) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  DataLayout GetPreferredLayout() const override { return preferred_data_layout_; }

  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  // WebGPU EP disallow concurrent run because actual implementation (eg. WebGPU backend) relies on global states to
  // work, and concurrent run with async function may mess up the states and cause undefined behavior.
  bool ConcurrentRunSupported() const override { return false; }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;
  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

  // WebGPU EP reuses the Device ID as the key to get the WebGpuContext instance.
  int GetDeviceId() const override { return context_id_; }

  bool IsGraphCaptureEnabled() const override;
  bool IsGraphCaptured(int graph_annotation_id) const override;
  Status ReplayGraph(int graph_annotation_id) override;

 private:
  bool IsGraphCaptureAllowed() const;
  void IncrementRegularRunCountBeforeGraphCapture();
  int context_id_;
  const webgpu::WebGpuContext& context_;
  DataLayout preferred_data_layout_;
  bool enable_graph_capture_ = false;
  bool is_graph_captured_ = false;
  int regular_run_count_before_graph_capture_ = 0;
  const int min_num_runs_before_cuda_graph_capture_ = 1;  // required min regular runs before graph capture for the necessary memory allocations.
};

}  // namespace onnxruntime
