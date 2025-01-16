// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <libloaderapi.h>
#include <set>
#else
#include <dlfcn.h>
#endif

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "QnnLog.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/common/path_string.h"
#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class QnnModel;

// configuration values for QnnBackendManager creation
struct QnnBackendManagerConfig {
  std::string backend_path;
  ProfilingLevel profiling_level_etw;
  ProfilingLevel profiling_level;
  std::string profiling_file_path;
  ContextPriority context_priority;
  std::string qnn_saver_path;
  uint32_t device_id;
  QnnHtpDevice_Arch_t htp_arch;
  uint32_t soc_model;
  bool enable_htp_weight_sharing;
};

class QnnBackendManager : public std::enable_shared_from_this<QnnBackendManager> {
 private:
  // private tag to pass to constructor to ensure that constructor cannot be directly called externally
  struct PrivateConstructorTag {};

 public:
  static std::shared_ptr<QnnBackendManager> Create(const QnnBackendManagerConfig& config) {
    return std::make_shared<QnnBackendManager>(config, PrivateConstructorTag{});
  }

  // Note: Creation should be done via Create(). This constructor is public so that it can be called from
  // std::make_shared().
  QnnBackendManager(const QnnBackendManagerConfig& config, PrivateConstructorTag)
      : backend_path_(config.backend_path),
        profiling_level_etw_(config.profiling_level_etw),
        profiling_level_(config.profiling_level),
        profiling_file_path_(config.profiling_file_path),
        context_priority_(config.context_priority),
        qnn_saver_path_(config.qnn_saver_path),
        device_id_(config.device_id),
        htp_arch_(config.htp_arch),
        soc_model_(config.soc_model),
        enable_htp_weight_sharing_(config.enable_htp_weight_sharing) {
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnBackendManager);

  ~QnnBackendManager();

  std::unique_ptr<unsigned char[]> GetContextBinaryBuffer(uint64_t& written_buffer_size);

  Status LoadCachedQnnContextFromBuffer(char* buffer, uint64_t buffer_length,
                                        std::string node_name,
                                        std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models,
                                        int64_t max_spill_fill_size);

  // Initializes handles to QNN resources (device, logger, etc.).
  // NOTE: This function locks the internal `logger_recursive_mutex_`.
  Status SetupBackend(const logging::Logger& logger, bool load_from_cached_context, bool need_load_system_lib);

  Status CreateHtpPowerCfgId(uint32_t deviceId, uint32_t coreId, uint32_t& htp_power_config_id);

  Status SetHtpPowerConfig(uint32_t htp_power_config_client_id,
                           HtpPerformanceMode htp_performance_mode);

  Status SetRpcControlLatency(uint32_t htp_power_config_client_id,
                              uint32_t rpc_control_latency);

  const QNN_INTERFACE_VER_TYPE& GetQnnInterface() { return qnn_interface_; }

  const Qnn_ContextHandle_t& GetQnnContext(int index = 0) {
    ORT_ENFORCE((contexts_.size() > 0) && (static_cast<size_t>(index) < contexts_.size()), "No valid QNN context!");
    return contexts_[index];
  }

  size_t GetQnnContextSize() {
    return contexts_.size();
  }

  const Qnn_BackendHandle_t& GetQnnBackendHandle() { return backend_handle_; }

  const Qnn_ProfileHandle_t& GetQnnProfileHandle() { return profile_backend_handle_; }

  // Resets the QNN log level to the given ORT log level or to the default log level if the argument is
  // std::nullopt.
  // NOTE: This function locks the internal `logger_recursive_mutex_`.
  Status ResetQnnLogLevel(std::optional<logging::Severity> ort_log_level = std::nullopt);

  Status ExtractBackendProfilingInfo();
  Status ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id, std::ofstream& outfile,
                                   bool backendSupportsExtendedEventData, bool tracelogging_provider_ep_enabled);
  Status ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id, const std::string& eventLevel,
                               std::ofstream& outfile, bool backendSupportsExtendedEventData,
                               bool tracelogging_provider_ep_enabled);

  Status SetProfilingLevelETW(ProfilingLevel profiling_level_etw_param);

  void SetQnnBackendType(uint32_t backend_id);
  QnnBackendType GetQnnBackendType() { return qnn_backend_type_; }

  const std::string& GetSdkVersion() { return sdk_build_version_; }

  Status DestroyHTPPowerConfigID(uint32_t htp_power_config_id);

  Status GetMaxSpillFillBufferSize(unsigned char* buffer,
                                   uint64_t buffer_length,
                                   uint64_t& max_spill_fill_buffer_size);

  // Gets an existing QNN mem handle or registers a new one.
  // `mem_handle` is set to the QNN mem handle.
  Status GetOrRegisterContextMemHandle(Qnn_ContextHandle_t context, void* shared_memory_address,
                                       const Qnn_Tensor_t& qnn_tensor,
                                       Qnn_MemHandle_t& mem_handle);

 private:
  Status LoadBackend();

  Status InitializeBackend();

  Status CreateDevice();

  Status ReleaseDevice();

  Status ShutdownBackend();

  Status InitializeProfiling();

  Status ReleaseProfilehandle();

  Status CreateContext();

  Status ReleaseContext();

  // Sets the ORT logger and creates a corresponding QNN logger with the same log level.
  // NOTE: caller must lock the `logger_recursive_mutex_` before calling this function.
  Status InitializeQnnLog(const logging::Logger& logger);

  // Terminate logging in the backend
  // NOTE: This function locks the internal `logger_recursive_mutex_`.
  Status TerminateQnnLog();

  // Releases all QNN resources. Called in the destructor.
  // NOTE: This function indirectly locks the internal `logger_recursive_mutex_` via nested function calls.
  void ReleaseResources();

  void* LoadLib(const char* file_name, int flags, std::string& error_msg);

  Status LoadQnnSystemLib();

  Status LoadQnnSaverBackend();

  Status UnloadLib(void* handle);

  void* LibFunction(void* handle, const char* symbol, std::string& error_msg);

  template <class T>
  inline T ResolveSymbol(void* lib_handle, const char* sym, const logging::Logger& logger) {
    std::string error_msg = "";
    T ptr = (T)LibFunction(lib_handle, sym, error_msg);
    if (ptr == nullptr) {
      LOGS(logger, ERROR) << "Unable to access symbol: " << sym << ". error: " << error_msg;
    }
    return ptr;
  }

  template <typename F, class T>
  Status GetQnnInterfaceProvider(const char* lib_path,
                                 const char* interface_provider_name,
                                 void** backend_lib_handle,
                                 Qnn_Version_t req_version,
                                 T** interface_provider);

  bool IsDevicePropertySupported();

  template <typename T>
  std::vector<std::add_pointer_t<std::add_const_t<T>>> ObtainNullTermPtrVector(const std::vector<T>& vec) {
    std::vector<std::add_pointer_t<std::add_const_t<T>>> ret;
    for (auto& elem : vec) {
      ret.push_back(&elem);
    }
    ret.push_back(nullptr);
    return ret;
  }

  std::string GetBackendBuildId() {
    char* backend_build_id{nullptr};
    if (QNN_SUCCESS != qnn_interface_.backendGetBuildId((const char**)&backend_build_id)) {
      LOGS(*logger_, ERROR) << "Unable to get build Id from the backend.";
    }
    return (backend_build_id == nullptr ? std::string("") : std::string(backend_build_id));
  }

  Status ExtractProfilingEventBasic(QnnProfile_EventId_t profile_event_id, const std::string& eventLevel,
                                    std::ofstream& outfile, bool tracelogging_provider_ep_enabled);
  Status ExtractProfilingEventExtended(QnnProfile_EventId_t profile_event_id, const std::string& eventLevel,
                                       std::ofstream& outfile, bool tracelogging_provider_ep_enabled);
  static const std::string& GetUnitString(QnnProfile_EventUnit_t unitType);
  static const std::unordered_map<QnnProfile_EventUnit_t, std::string>& GetUnitStringMap();
  static const std::string GetEventTypeString(QnnProfile_EventType_t eventType);
  static const std::string ExtractQnnScalarValue(const Qnn_Scalar_t& scalar);
  const char* QnnProfileErrorToString(QnnProfile_Error_t error);
  std::string_view QnnErrorHandleToString(Qnn_ErrorHandle_t error);
  QnnLog_Level_t MapOrtSeverityToQNNLogLevel(logging::Severity ort_log_level);
#ifdef _WIN32
  void LogQnnProfileEventAsTraceLogging(
      uint64_t timestamp,
      const std::string& message,
      const std::string& qnnScalarValue,
      const std::string& unit,
      const std::string& timingSource,
      const std::string& eventLevel,
      const char* eventIdentifier);
#endif

  // Adds a new QNN context.
  // Transfers ownership of `context_handle` (i.e., responsibility of freeing it) to this instance.
  Status AddQnnContextHandle(Qnn_ContextHandle_t context_handle);

 private:
  // assume Qnn_ContextHandle_t is a pointer and able to be wrapped with std::unique_ptr
  static_assert(std::is_pointer_v<Qnn_ContextHandle_t>);
  using UniqueQnnContextHandle =
      std::unique_ptr<std::remove_pointer_t<Qnn_ContextHandle_t>, std::function<void(Qnn_ContextHandle_t)>>;

  struct QnnContextHandleRecord {
    UniqueQnnContextHandle context_handle;
    std::unique_ptr<QnnContextMemHandleManager> mem_handles;
  };

 private:
  const std::string backend_path_;
  std::recursive_mutex logger_recursive_mutex_;
  const logging::Logger* logger_ = nullptr;
  QNN_INTERFACE_VER_TYPE qnn_interface_ = QNN_INTERFACE_VER_TYPE_INIT;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_sys_interface_ = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
  void* backend_lib_handle_ = nullptr;
  void* system_lib_handle_ = nullptr;
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  QnnBackend_Config_t** backend_config_ = nullptr;
  Qnn_LogHandle_t log_handle_ = nullptr;
  Qnn_DeviceHandle_t device_handle_ = nullptr;

  // Map of Qnn_ContextHandle_t to QnnContextHandleRecord.
  // The QnnContextHandleRecord has ownership of the Qnn_ContextHandle_t.
  // Note: Using shared_ptr<QnnContextHandleRecord> so that we can refer to it with a weak_ptr from a
  // HtpSharedMemoryAllocator allocation cleanup callback.
  std::unordered_map<Qnn_ContextHandle_t, std::shared_ptr<QnnContextHandleRecord>> context_map_;

  // Vector of Qnn_ContextHandle_t. The context handles are owned by context_map_.
  std::vector<Qnn_ContextHandle_t> contexts_;

  ProfilingLevel profiling_level_etw_;
  ProfilingLevel profiling_level_;
  ProfilingLevel profiling_level_merge_;
  const std::string profiling_file_path_;
  bool backend_initialized_ = false;
  bool device_created_ = false;
  bool context_created_ = false;
  bool backend_setup_completed_ = false;
  // NPU backend requires quantized model
  QnnBackendType qnn_backend_type_ = QnnBackendType::CPU;
  Qnn_ProfileHandle_t profile_backend_handle_ = nullptr;
  std::vector<std::string> op_package_paths_;
  ContextPriority context_priority_;
  std::string sdk_build_version_ = "";
#ifdef _WIN32
  std::set<HMODULE> mod_handles_;
#endif
  const std::string qnn_saver_path_;
  uint32_t device_id_ = 0;
  QnnHtpDevice_Arch_t htp_arch_ = QNN_HTP_DEVICE_ARCH_NONE;
  uint32_t soc_model_ = QNN_SOC_MODEL_UNKNOWN;
  bool enable_htp_weight_sharing_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
