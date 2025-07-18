// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "HTP/QnnHtpGraph.h"

#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/builder/qnn_configs_helper.h"
#include "core/providers/qnn-abi/builder/qnn_def.h"
#include "core/providers/qnn-abi/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn-abi/builder/qnn_model.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "test/autoep/library/example_plugin_ep_utils.h"

namespace onnxruntime {
class QnnEpFactory;

// Forward declaration for QnnBackendManager
namespace qnn {
class QnnBackendManager;
}

class QnnEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context{false};
    bool share_ep_contexts{false};
    bool enable_vtcm_backup_buffer_sharing{false};
    bool disable_cpu_ep_fallback{false};
  };

  QnnEp(QnnEpFactory& factory, const std::string& name,
        const Config& config, const OrtLogger& logger);
  ~QnnEp();

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                  const OrtGraph* graph,
                                                  OrtEpGraphSupportInfo* graph_support_info);
  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr,
                                             _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes,
                                             _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes);

  OrtStatus* GetSupportedNodes(OrtEp* this_ptr,
                               const OrtGraph* graph,
                               const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                               const size_t node_unit_size,
                               const logging::Logger& logger,
                               std::vector<const OrtNode*>& supported_nodes) const;

  // // Helper functions
  // int GenerateMetadefId(const OrtGraph* graph, uint64_t& model_hash);
  // std::string MakeMetadefName(const OrtGraph* graph);
  // bool EpSharedContextsHasAllGraphs(const OrtGraph* graph);
  void PartitionCtxModel(const OrtEp* this_ptr, const OrtGraph* graph, size_t num_nodes_in_graph,
                         OrtEpGraphSupportInfo* graph_support_info);
  // static void GetMainEPCtxNodes(QnnEp* ep, const OrtGraph* graph, std::unordered_set<const OrtNode*>& ep_context_nodes);
  // void GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
  //                                  const std::string& model_path_string,
  //                                  std::string& context_model_path);

  // // Run start/end methods
  // OrtStatus* OnRunStart(const OrtGraph* graph, const OrtRunOptions* run_options);
  // OrtStatus* OnRunEnd(const OrtGraph* graph, const OrtRunOptions* run_options, bool sync_stream);

  void InitQnnHtpGraphConfigs(
      qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const;

  // Per-thread context management
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

  using PerThreadContextMap = std::unordered_map<const QnnEp*, std::weak_ptr<PerThreadContext>>;

  struct ContextCacheHolder {
    ContextCacheHolder() {
      // Note: This would typically use RunOnUnload in the full implementation
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
    // weak references to thread local caches from which this QnnEp instance's entry should be removed
    // upon destruction
    std::set<std::weak_ptr<PerThreadContextMap>, std::owner_less<std::weak_ptr<PerThreadContextMap>>>
        caches_to_update_on_destruction;
    // synchronizes access to PerThreadContextState members
    std::mutex mutex;
  };

  // The execution provider maintains the PerThreadContexts in this structure.
  mutable PerThreadContextState context_state_;

  PerThreadContext& GetPerThreadContext();
  void ReleasePerThreadContext();

  const QnnEpFactory& factory_;
  std::string name_;
  Config config_;
  const OrtLogger& logger_;
  bool context_cache_enabled_;
  bool share_ep_contexts_;
  bool enable_vtcm_backup_buffer_sharing_;
  std::string context_node_name_prefix_;
  std::string context_cache_path_cfg_;

  // Metadef ID generation state
  mutable std::unordered_map<uint64_t, uint64_t> main_graph_hash_;
  mutable std::unordered_map<uint64_t, int> model_metadef_id_;

  // QNN-related.
  std::shared_ptr<qnn::QnnBackendManager> qnn_backend_manager_;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models_;

  // Configurations for HTP backend.
  uint32_t device_id_{0};
  qnn::HtpPerformanceMode default_htp_performance_mode_{qnn::HtpPerformanceMode::kHtpDefault};
  uint32_t default_rpc_control_latency_{0};
  qnn::ModelSettings model_settings_ = {};
  qnn::HtpGraphFinalizationOptimizationMode htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  int32_t vtcm_size_in_mb_ = 0;
  bool enable_HTP_FP16_precision_ = true;

  bool dump_json_qnn_graph_ = false;
  std::string json_qnn_graph_dir_ = "";
};

}
