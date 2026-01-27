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
#include "core/providers/qnn-abi/builder/qnn_cache_compatibility_manager.h"
#include "core/providers/qnn-abi/builder/qnn_configs_helper.h"
#include "core/providers/qnn-abi/builder/qnn_def.h"
#include "core/providers/qnn-abi/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn-abi/builder/qnn_def.h"
#include "core/providers/qnn-abi/builder/qnn_model.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn-abi/qnn_telemetry.h"
#include "core/providers/qnn-abi/rpcmem_library.h"

namespace onnxruntime {
class QnnEpFactory;

// Forward declaration for QnnBackendManager
namespace qnn {
class QnnBackendManager;
}

class QnnEp : public OrtEp, public ApiPtrs {
 public:
  QnnEp(QnnEpFactory& factory,
        const std::string& name,
        const OrtSessionOptions& session_options,
        const OrtLogger* logger);
  ~QnnEp();

  OrtStatus* ValidateCompiledModelCompatibilityInfo(const OrtHardwareDevice* const* devices,
                                                    size_t num_devices,
                                                    const char* compatibility_info,
                                                    OrtCompiledModelCompatibility* model_compatibility) noexcept;

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                   const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr,
                                             _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes,
                                             _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;
  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                            _Out_ OrtEpDataLayout* preferred_data_layout) noexcept;
  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                                  _In_z_ const char* domain,
                                                                  _In_z_ const char* op_type,
                                                                  _In_ OrtEpDataLayout target_data_layout,
                                                                  _Outptr_ int* should_convert) noexcept;
  static OrtStatus* ORT_API_CALL OnRunStartImpl(_In_ OrtEp* this_ptr, _In_ const ::OrtRunOptions* run_options) noexcept;
  static OrtStatus* ORT_API_CALL OnRunEndImpl(_In_ OrtEp* this_ptr,
                                              _In_ const ::OrtRunOptions* run_options,
                                              _In_ bool sync_stream) noexcept;
  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;
  static OrtStatus* ORT_API_CALL SetDynamicOptionsImpl(_In_ OrtEp* this_ptr,
                                                       _In_reads_(num_options) const char* const* option_keys,
                                                       _In_reads_(num_options) const char* const* option_values,
                                                       _In_ size_t num_options) noexcept;
  static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(_In_ OrtEp* this_ptr,
                                                                        _In_ const OrtGraph* graph) noexcept;

  OrtStatus* GetSupportedNodes(const OrtGraph* graph,
                               const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                               const size_t node_unit_size,
                               std::vector<const OrtNode*>& supported_nodes) const;

  void PartitionCtxModel(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info);

  OrtStatus* CompileContextModel(const OrtGraph** graphs,
                                 const OrtNode** fused_nodes,
                                 size_t count,
                                 OrtNodeComputeInfo** node_compute_infos);

  OrtStatus* CreateEPContextNodes(const OrtGraph* graph,
                                  const OrtNode** fused_nodes,
                                  size_t count,
                                  OrtNode** ep_context_nodes);

  // // Helper functions
  // int GenerateMetadefId(const OrtGraph* graph, uint64_t& model_hash);
  // std::string MakeMetadefName(const OrtGraph* graph);
  void ParseHtpGraphFinalizationOptimizationMode(const std::string& htp_graph_finalization_opt_mode_string,
                                                 const Ort::Logger& logger);

  bool IsHtpSharedMemoryAllocatorAvailable() const { return rpcmem_library_ != nullptr; }

  void InitQnnHtpGraphConfigs(
      qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const;
  std::unique_ptr<qnn::QnnSerializerConfig> InitQnnSerializerConfig() const;

  std::string FormatEPConfigKey(const std::string& key) const {
    return GetProviderOptionPrefix(name_) + key;
  }

  struct QnnNodeComputeInfo : OrtNodeComputeInfo {
    explicit QnnNodeComputeInfo(QnnEp& ep);

    static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* compute_context,
                                                   void** compute_state);
    static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                               OrtKernelContext* kernel_context);
    static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

    QnnEp& ep;
  };

  // Will return true if any power config options need to be updated
  bool GetPerThreadHtpPowerConfigs(qnn::PerThreadHtpPowerConfigs_t& per_thread_htp_power_configs,
                                   const ::OrtRunOptions* run_options);

  void CreateHtpPowerConfigId() const;
  // Will return false if htp_power_config_id_ has no value
  bool GetHtpPowerConfigId(uint32_t& htp_power_config_id);

  // htp_power_config_id_ must be created during GetCapability() as it is the only
  // step during setup/initialization in which QNNBackendManager is setup and ready.
  // GetCapability() is a const function, so these options must be mutable
  mutable std::optional<uint32_t> htp_power_config_id_;
  mutable std::mutex config_id_mutex_;

  // const QnnEpFactory& factory_;
  std::string name_;
  const Ort::Logger logger_;
  bool context_cache_enabled_ = false;
  bool share_ep_contexts_ = false;
  bool enable_vtcm_backup_buffer_sharing_ = false;
  std::string context_node_name_prefix_ = "";
  std::string context_cache_path_cfg_ = "";
  const OrtSessionOptions& session_options_;

  bool disable_cpu_ep_fallback_ = false;  // True if CPU EP fallback has been disabled for this session.
  bool qnn_context_embed_mode_ = true;
  bool stop_share_ep_contexts_ = false;
  bool enable_spill_fill_buffer_ = false;
#if defined(_WIN32)
  qnn::QnnTelemetry::EtwInternalCallback callback_ETWSink_provider_ = nullptr;
#endif

  // Metadef ID generation state
  mutable std::unordered_map<uint64_t, uint64_t> main_graph_hash_;
  mutable std::unordered_map<uint64_t, int> model_metadef_id_;

  // QNN-related.
  std::shared_ptr<qnn::QnnBackendManager> qnn_backend_manager_;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models_;

  // Configurations for HTP backend.
  uint32_t device_id_{0};
  qnn::HtpPerformanceMode default_htp_performance_mode_{qnn::HtpPerformanceMode::kHtpDefault};
  uint32_t default_rpc_control_latency_ = 0;
  uint32_t default_rpc_polling_time_ = 0;
  qnn::ModelSettings model_settings_ = {};
  qnn::HtpGraphFinalizationOptimizationMode htp_graph_finalization_opt_mode_ = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  int32_t vtcm_size_in_mb_ = 0;
  bool enable_HTP_FP16_precision_ = true;

  bool dump_json_qnn_graph_ = false;
  std::string json_qnn_graph_dir_ = "";

  // Whether this is set depends on a session option enabling it and if the RPCMEM dynamic library is available.
  // This is potentially shared with HtpSharedMemoryAllocator which may be returned by CreatePreferredAllocators().
  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library_ = nullptr;

  // Model compatibility.
  std::shared_ptr<qnn::QnnCacheCompatibilityManager> qnn_cache_compatibility_manager_ = nullptr;
  qnn::QnnCompatibilityInfo compatibility_info_;
  // Format: <BackendId>:<SDK>:<BackendApi>:<ContextBlob>:<HtpArch>:<IsHtpUsrDrv>.
  std::string compatibility_info_string_ = "";
};

}  // namespace onnxruntime
