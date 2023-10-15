// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include <string>
#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"

namespace onnxruntime {

// Logical device representation.
class QNNExecutionProvider : public IExecutionProvider {
 public:
  explicit QNNExecutionProvider(const ProviderOptions& provider_options_map, const SessionOptions* session_options);
  virtual ~QNNExecutionProvider() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QNNExecutionProvider);

  // we implement the Compile that takes FusedNodeAndGraph instances
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const IKernelLookup& /*kernel_lookup*/) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }

  DataLayout GetPreferredLayout() const override;

 private:
  void ParseProfilingLevel(std::string profiling_level_string);

  bool IsNodeSupported(qnn::QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                       std::unordered_map<const NodeUnit*, bool>& node_unit_supported_result,
                       const logging::Logger& logger) const;

  std::unordered_set<const Node*> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                    const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                                    const size_t node_unit_size,
                                                    bool load_from_cached_context,
                                                    const logging::Logger& logger) const;

  Status CreateComputeFunc(std::vector<NodeComputeInfo>& node_compute_funcs,
                           const logging::Logger& logger);

  Status CompileFromOrtGraph(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                             std::vector<NodeComputeInfo>& node_compute_funcs,
                             const logging::Logger& logger);

  void ParseHtpPerformanceMode(std::string htp_performance_mode_string);

 private:
  ProviderOptions runtime_options_;
  qnn::ProfilingLevel profiling_level_ = qnn::ProfilingLevel::OFF;
  qnn::HtpPerformanceMode htp_performance_mode_ = qnn::HtpPerformanceMode::kHtpDefault;
  std::unique_ptr<qnn::QnnBackendManager> qnn_backend_manager_;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models_;
  uint32_t rpc_control_latency_ = 0;
  bool context_cache_enabled_ = false;
  std::string context_cache_path_ = "";
  bool disable_cpu_ep_fallback_ = false;  // True if CPU EP fallback has been disabled for this session.
  std::unique_ptr<qnn::QnnCacheModelHandler> qnn_cache_model_handler_;
};

}  // namespace onnxruntime
