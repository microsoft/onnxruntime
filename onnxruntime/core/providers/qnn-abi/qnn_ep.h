// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "core/providers/qnn-abi/builder/qnn_def.h"
#include "core/providers/qnn-abi/builder/onnx_ctx_model_helper.h"
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

  QnnEp(const QnnEpFactory& factory, const std::string& name,
        const Config& config, const OrtLogger* logger);
  ~QnnEp();

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                  const OrtGraph* graph,
                                                  OrtEpGraphSupportInfo* graph_support_info);

  // Helper functions
  int GenerateMetadefId(const OrtGraph* graph, uint64_t& model_hash);
  std::string MakeMetadefName(const OrtGraph* graph);
  bool EpSharedContextsHasAllGraphs(const OrtGraph* graph);
  void PartitionCtxModel(const OrtGraph* graph, size_t num_nodes_in_graph,
                        OrtEpGraphSupportInfo* graph_support_info);
  static void GetMainEPCtxNodes(QnnEp* ep, const OrtGraph* graph, std::unordered_set<const OrtNode*>& ep_context_nodes);
  void GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
                                   const std::string& model_path_string,
                                   std::string& context_model_path);

  // Run start/end methods
  OrtStatus* OnRunStart(const OrtGraph* graph, const OrtRunOptions* run_options);
  OrtStatus* OnRunEnd(const OrtGraph* graph, const OrtRunOptions* run_options, bool sync_stream);

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
  const OrtLogger* logger_;
  bool context_cache_enabled_;
  bool share_ep_contexts_;
  bool enable_vtcm_backup_buffer_sharing_;
  std::string context_node_name_prefix_;
  std::string context_cache_path_cfg_;

  // Metadef ID generation state
  mutable std::unordered_map<uint64_t, uint64_t> main_graph_hash_;
  mutable std::unordered_map<uint64_t, int> model_metadef_id_;

  // Backend manager for QNN operations
  std::shared_ptr<qnn::QnnBackendManager> qnn_backend_manager_;

  // Configuration for HTP backend
  uint32_t device_id_{0};
  qnn::HtpPerformanceMode default_htp_performance_mode_{qnn::HtpPerformanceMode::kHtpDefault};
  uint32_t default_rpc_control_latency_{0};
};

}
