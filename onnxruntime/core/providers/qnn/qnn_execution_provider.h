// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/graph/model.h"
#include <string>
#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_configs_helper.h"
#include "HTP/QnnHtpGraph.h"
#include <vector>
#include <set>
#include <unordered_map>

namespace onnxruntime {

void RunOnUnload(std::function<void()> function);

// Logical device representation.
class QNNExecutionProvider : public IExecutionProvider {
 public:
  explicit QNNExecutionProvider(const ProviderOptions& provider_options_map, const SessionOptions* session_options);
  virtual ~QNNExecutionProvider();
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

  const InlinedVector<const Node*> GetEpContextNodes() const override;

  Status OnRunStart(const onnxruntime::RunOptions& run_options) override;

  Status OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) override;

 private:
  bool IsNodeSupported(qnn::QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
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

  void ParseHtpGraphFinalizationOptimizationMode(const std::string& htp_graph_finalization_opt_mode_string);

  void InitQnnGraphConfigs(qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const;

 private:
  qnn::HtpGraphFinalizationOptimizationMode htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  std::unique_ptr<qnn::QnnBackendManager> qnn_backend_manager_;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models_;
  bool context_cache_enabled_ = false;
  std::string context_cache_path_cfg_ = "";
  bool disable_cpu_ep_fallback_ = false;  // True if CPU EP fallback has been disabled for this session.
  bool qnn_context_embed_mode_ = true;
  int32_t vtcm_size_in_mb_ = 0;
  std::unique_ptr<onnxruntime::Model> qnn_ep_context_model_;
  ModelMetadefIdGenerator metadef_id_generator_;
  uint32_t device_id_ = 0;
  qnn::HtpPerformanceMode default_htp_performance_mode_ = qnn::HtpPerformanceMode::kHtpDefault;
  uint32_t default_rpc_control_latency_ = 0;

  class PerThreadContext final {
   public:
    PerThreadContext(qnn::QnnBackendManager* qnn_backend_manager,
                     uint32_t device_id, uint32_t core_id,
                     qnn::HtpPerformanceMode default_htp_performance_mode,
                     uint32_t default_rpc_control_latency);
    ~PerThreadContext();
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PerThreadContext);

    bool IsHtpPowerConfigIdValid() { return is_htp_power_config_id_valid_; }

    uint32_t GetHtpPowerConfigId() { return htp_power_config_id_; }

   private:
    bool is_htp_power_config_id_valid_ = false;
    uint32_t htp_power_config_id_ = 0;
    qnn::QnnBackendManager* qnn_backend_manager_;
  };

  using PerThreadContextMap = std::unordered_map<const QNNExecutionProvider*, std::weak_ptr<PerThreadContext>>;

  struct ContextCacheHolder {
    ContextCacheHolder() {
      RunOnUnload([&, weak_p_ = std::weak_ptr<PerThreadContextMap>(p)] {
        if (auto lock = weak_p_.lock())
          p.reset();
      });
    }

    std::shared_ptr<PerThreadContextMap> p = std::make_shared<PerThreadContextMap>();
  };

  static const std::shared_ptr<PerThreadContextMap>& PerThreadContextCache() {
    thread_local const ContextCacheHolder per_thread_context_cache;
    return per_thread_context_cache.p;
  }

  struct PerThreadContextState {
    // contexts that are currently active
    std::set<std::shared_ptr<PerThreadContext>, std::owner_less<std::shared_ptr<PerThreadContext>>> active_contexts;
    // contexts available for reuse
    std::vector<std::shared_ptr<PerThreadContext>> retired_context_pool;
    // weak references to thread local caches from which this QNNExecutionProvider instance's entry should be removed
    // upon destruction
    std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
        caches_to_update_on_destruction;
    // synchronizes access to PerThreadContextState members
    OrtMutex mutex;
  };

  // The execution provider maintains the PerThreadContexts in this structure.
  // Synchronization is required to update the contained structures.
  // On the other hand, access to an individual PerThreadContext is assumed to be from a single thread at a time,
  // so synchronization is not required for that.
  mutable PerThreadContextState context_state_;

  PerThreadContext& GetPerThreadContext() const;
  void ReleasePerThreadContext() const;
};

}  // namespace onnxruntime
