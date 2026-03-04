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
#include <thread>
#include <unordered_map>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "QnnLog.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/rpcmem_library.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_htp_power_config_manager.h"
#include "core/providers/qnn/builder/qnn_profile_serializer.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
#include "core/providers/qnn/builder/qnn_file_mapping_interface.h"
#endif

namespace onnxruntime {
namespace qnn {

class QnnModel;

class QnnSerializerConfig {
 public:
  virtual ~QnnSerializerConfig();

  /**
   * Create a config to write a DLC file for each graph using the Ir backend.
   */
  static std::unique_ptr<QnnSerializerConfig> CreateIr(std::string backend_path, std::string dlc_dir);

  /**
   * Create a config to write C++ source files using the Saver backend.
   */
  static std::unique_ptr<QnnSerializerConfig> CreateSaver(std::string backend_path);

  /**
   * Get the path to the serializer backend.
   */
  const std::string& GetBackendPath() const;

  /**
   * Set the name of the graph being serialized. This value may be used to determine the name
   * of the output files.
   *
   * \param graph_name The name of the graph being serialized.
   */
  void SetGraphName(std::string graph_name);

  /**
   * Gets the name of the graph being serialized.
   *
   * \return graph_name The name of the graph being serialized.
   */
  const std::string& GetGraphName() const;

  /**
   * Get any QNN Graph configs required to configure this serializer and perform any
   * preparation, such as creating output directories.
   *
   * \return nullptr or a null-terminated list of QnnGraph_Config_t*.
   */
  virtual const QnnGraph_Config_t** Configure() = 0;

  /**
   * Some serializers allow for GraphConfigs that are unrelated to serialization to be
   * specified at context creation time, while others raise an error. If true, this
   * serializer should be configured with graph configs for any applicable real (e.g., HTP)
   * backend.
   *
   * \return true if the backend can be configured with non-serialization graph configs.
   */
  virtual bool SupportsArbitraryGraphConfigs() const = 0;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnSerializerConfig);

 protected:
  QnnSerializerConfig(std::string backend_path);

 private:
  std::string backend_path_;
  std::string graph_name_{"graph"};
};

struct OpPackage {
  std::string op_type;
  std::string path;
  std::string interface;
  std::string target;
};

// configuration values for QnnBackendManager creation
struct QnnBackendManagerConfig {
  std::string backend_path;
  ProfilingLevel profiling_level_etw;
  ProfilingLevel profiling_level;
  std::string profiling_file_path;
  ContextPriority context_priority;
  std::shared_ptr<QnnSerializerConfig> qnn_serializer_config;
  uint32_t device_id;
  QnnHtpDevice_Arch_t htp_arch;
  uint32_t soc_model;
  std::vector<OpPackage> op_packages;
  bool skip_qnn_version_check;
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
        qnn_serializer_config_(config.qnn_serializer_config),
        device_id_(config.device_id),
        htp_arch_(config.htp_arch),
        soc_model_(config.soc_model),
        op_packages_(config.op_packages),
        skip_qnn_version_check_(config.skip_qnn_version_check) {
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnBackendManager);

  ~QnnBackendManager();

  std::unique_ptr<unsigned char[]> GetContextBinaryBuffer(uint64_t& written_buffer_size);

  Status LoadCachedQnnContextFromBuffer(char* buffer, uint64_t buffer_length,
                                        const std::string& context_bin_filepath,
                                        std::string node_name,
                                        std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models,
                                        int64_t max_spill_fill_size);

  // Initializes handles to QNN resources (device, logger, etc.).
  // NOTE: This function locks the internal `logger_recursive_mutex_`.
  Status SetupBackend(const logging::Logger& logger, bool load_from_cached_context,
                      bool need_load_system_lib, bool share_ep_contexts,
                      bool enable_vtcm_backup_buffer_sharing,
                      bool enable_file_mapped_weights,
                      std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library,
                      std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>& context_bin_map,
                      bool enable_htp_extended_udma_mode);

  Status CreateHtpPowerCfgId(uint32_t deviceId, uint32_t coreId, uint32_t& htp_power_config_id);

  Status SetHtpPowerConfigs(uint32_t htp_power_config_client_id,
                            HtpPerformanceMode htp_performance_mode,
                            uint32_t rpc_polling_time,
                            uint32_t rpc_control_latency);

  Status SetPerThreadHtpPowerConfigs(const std::thread::id& thread_id, bool pre_run);

  Status AddPerThreadHtpPowerConfigMapping(const std::thread::id& thread_id,
                                           const PerThreadHtpPowerConfigs_t& htp_power_configs);

  void RemovePerThreadHtpPowerConfigMapping(const std::thread::id& thread_id);

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

  Status ExtractBackendProfilingInfo(qnn::profile::ProfilingInfo& profiling_info);

  Status ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id, profile::Serializer& profile_writer,
                                   bool backendSupportsExtendedEventData);

  Status ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id, const std::string& eventLevel,
                               profile::Serializer& profile_writer, bool backendSupportsExtendedEventData);

  Status SetProfilingLevelETW(ProfilingLevel profiling_level_etw_param);

  void SetQnnBackendType(uint32_t backend_id);
  QnnBackendType GetQnnBackendType() { return qnn_backend_type_; }

  uint32_t GetSocModel() const { return soc_model_; }

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

  Status ParseLoraConfig(std::string lora_config);

  QnnSerializerConfig* GetQnnSerializerConfig();

  // Handler to be called upon successful context creation via contextCreateFromBinaryListAsync()
  // This handler is expected to be called in the callback ContextCreateAsyncCallback() in the .cc file
  // Takes in the context and the notifyParam objects received by the callback function
  // notifyParam is expected to be a pointer to a vector of node names associated with that context handle
  // For each node name, a mapping to the context handle will be created
  void ProcessContextFromBinListAsync(Qnn_ContextHandle_t handle, void* notifyParam);

  // Sets the context priority to the given value, if valid
  Status SetContextPriority(ContextPriority context_priority);
  // Resets the context priority to the session default as defined by context_priority_
  Status ResetContextPriority();

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  bool ProfilingEnabled() { return profiling_enabled_; }
#endif

  bool FileMappingIsEnabled() {
    return file_mapped_weights_enabled_;
  }

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  Qnn_ErrorHandle_t MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                               Qnn_ContextBinaryDmaDataResponse_t* response,
                               void* const mapped_base_ptr,
                               const size_t file_size);

  Qnn_ErrorHandle_t ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem, void* mapped_base_ptr);
#endif

  QnnLog_Level_t MapOrtSeverityToQNNLogLevel(logging::Severity ort_log_level);
  static logging::Severity MapQNNLogLevelToOrtSeverity(QnnLog_Level_t qnn_log_level);

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  typedef struct FileMappingCallbackInfo {
    void* const mapped_file_ptr;
    const size_t file_size;
    QnnBackendManager* const backend_manager;

    FileMappingCallbackInfo(void* ptr, size_t size, QnnBackendManager* manager)
        : mapped_file_ptr(ptr), file_size(size), backend_manager(manager) {}

  } FileMappingCallbackInfo_t;
#endif

 private:
  Status LoadBackend();

  Status InitializeBackend();

  Status CreateDevice();

  Status ReleaseDevice();

  Status ShutdownBackend();

  Status InitializeProfiling();

  Status ReleaseProfilehandle();

  Status CreateContext(bool enable_htp_weight_sharing, bool enable_htp_extended_udma_mode);

  Status GetFileSizeIfValid(const std::string& filepath, size_t& file_size);

  Status ReadContextBinIfValid(const std::string& context_bin_filepath,
                               std::vector<char>& buffer);

  Status CreateContextVtcmBackupBufferSharingEnabled(std::unordered_map<std::string,
                                                                        std::unique_ptr<std::vector<std::string>>>& context_bin_map);

  Status CreateContextFromListAsync(const QnnContext_Config_t** configs,
                                    std::unordered_map<std::string,
                                                       std::unique_ptr<std::vector<std::string>>>& context_bin_map);

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  Status CreateContextFromListAsyncWithCallback(const QnnContext_Config_t** configs,
                                                std::unordered_map<std::string,
                                                                   std::unique_ptr<std::vector<std::string>>>& context_bin_map);
#endif

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

  Status LoadQnnSerializerBackend();

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

  std::string GetBackendBuildId() {
    char* backend_build_id{nullptr};
    if (QNN_SUCCESS != qnn_interface_.backendGetBuildId((const char**)&backend_build_id)) {
      LOGS(*logger_, ERROR) << "Unable to get build Id from the backend.";
    }
    return (backend_build_id == nullptr ? std::string("") : std::string(backend_build_id));
  }

  Status ExtractProfilingEventBasic(QnnProfile_EventId_t profile_event_id, const std::string& eventLevel,
                                    profile::Serializer& profile_writer);

  Status ExtractProfilingEventExtended(QnnProfile_EventId_t profile_event_id, const std::string& eventLevel,
                                       profile::Serializer& profile_writer);

  const char* QnnProfileErrorToString(QnnProfile_Error_t error);
  std::string QnnErrorHandleToString(Qnn_ErrorHandle_t error);

  // Adds a new QNN context.
  // Transfers ownership of `context_handle` (i.e., responsibility of freeing it) to this instance
  Status AddQnnContextHandle(Qnn_ContextHandle_t context_handle);

  bool GetPerThreadHtpPowerConfigMapping(const std::thread::id& thread_id,
                                         PerThreadHtpPowerConfigs_t& htp_power_configs);

 private:
  // assume Qnn_ContextHandle_t is a pointer and able to be wrapped with std::unique_ptr
  static_assert(std::is_pointer_v<Qnn_ContextHandle_t>);
  using UniqueQnnContextHandle =
      std::unique_ptr<std::remove_pointer_t<Qnn_ContextHandle_t>, std::function<void(Qnn_ContextHandle_t)>>;

  struct QnnContextHandleRecord {
    UniqueQnnContextHandle context_handle;
    std::unique_ptr<QnnContextMemHandleManager> mem_handles;
  };

  Status LoadOpPackage() {
    // assume op_packages passed in represented in
    // op_packages|<OpTpye>:<PackagePath>:<InterfaceSymbolName>:<OptionalTarget>,<OpTpye2>:<PackagePath2>:<InterfaceSymbolName2>:<OptionalTarget2>
    for (const auto& op_package : op_packages_) {
      ORT_RETURN_IF(nullptr == qnn_interface_.backendRegisterOpPackage, "backendRegisterOpPackageFnHandle is nullptr.");

      Qnn_ErrorHandle_t result = qnn_interface_.backendRegisterOpPackage(
          backend_handle_,
          op_package.path.c_str(),
          op_package.interface.c_str(),
          op_package.target.c_str());

      if (result != QNN_SUCCESS) {
        switch (result) {
          case QNN_BACKEND_ERROR_INVALID_ARGUMENT:
            LOGS(*logger_, ERROR) << "Invalid argument, please check if op package path or interface provider is NULL.";
            break;
          case QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND:
            LOGS(*logger_, ERROR) << "Could not open op package path. op_pack_path: " << op_package.path;
            break;
          case QNN_BACKEND_ERROR_OP_PACKAGE_IF_PROVIDER_NOT_FOUND:
            LOGS(*logger_, ERROR) << "Could not find interfaceProvider symbol in op package library.";
            break;
          case QNN_BACKEND_ERROR_OP_PACKAGE_REGISTRATION_FAILED:
            LOGS(*logger_, ERROR) << "Op package registration failed.";
            break;
          case QNN_BACKEND_ERROR_OP_PACKAGE_UNSUPPORTED_VERSION:
            LOGS(*logger_, ERROR) << "Op package has interface version not supported by this backend.";
            break;
          case QNN_BACKEND_ERROR_NOT_SUPPORTED:
            LOGS(*logger_, ERROR) << "Op package registration is not supported.";
            break;
          case QNN_BACKEND_ERROR_INVALID_HANDLE:
            LOGS(*logger_, ERROR) << "backend is not a valid handle.";
            break;
          case QNN_BACKEND_ERROR_OP_PACKAGE_DUPLICATE:
            LOGS(*logger_, ERROR) << "OpPackageName+OpName must be unique. Op package content information can be be obtained with \
  QnnOpPackage interface. Indicates that an Op with the same package name and op name was already registered.";
            break;
          case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
            LOGS(*logger_, ERROR) << "SSR occurrence (successful recovery).";
            break;
          case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL:
            LOGS(*logger_, ERROR) << "SSR occurrence (unsuccessful recovery).";
            break;
          default:
            LOGS(*logger_, ERROR) << "Unknown error occurred while initializing logging in the QNN backend.";
            break;
        }
      }
      ORT_RETURN_IF(QNN_SUCCESS != result, "Failed to register op package to backend. Error: ", QnnErrorHandleToString(result));
      LOGS(*logger_, VERBOSE) << "Successfully register the op package.";
      std::string op_package_for_registration = op_package.interface;
      std::string suffix = "InterfaceProvider";
      if (op_package_for_registration.size() >= suffix.size() &&
          op_package_for_registration.compare(op_package_for_registration.size() - suffix.size(), suffix.size(), suffix) == 0) {
        op_package_for_registration.erase(op_package_for_registration.size() - suffix.size());
      }
      registerUDO(op_package.op_type, op_package_for_registration);
    }

    return Status::OK();
  }

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
  power::HtpPowerConfigManager htp_power_config_manager_;

  // Map of Qnn_ContextHandle_t to QnnContextHandleRecord.
  // The QnnContextHandleRecord has ownership of the Qnn_ContextHandle_t.
  // Note: Using shared_ptr<QnnContextHandleRecord> so that we can refer to it with a weak_ptr from a
  // HtpSharedMemoryAllocator allocation cleanup callback.
  std::unordered_map<Qnn_ContextHandle_t, std::shared_ptr<QnnContextHandleRecord>> context_map_;

  // Map of EP Main Context Node names to Qnn_ContextHandle_t
  std::mutex ep_context_handle_map_mutex_;
  std::unordered_map<std::string, Qnn_ContextHandle_t> ep_context_handle_map_;

  // Vector of Qnn_ContextHandle_t. The context handles are owned by context_map_.
  std::vector<Qnn_ContextHandle_t> contexts_;

  ProfilingLevel profiling_level_etw_;
  ProfilingLevel profiling_level_;
  ProfilingLevel profiling_level_merge_;
  const std::string profiling_file_path_;
  bool system_lib_loaded_ = false;

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  bool profiling_enabled_ = false;
#endif

  bool backend_initialized_ = false;
  bool device_created_ = false;
  bool context_created_ = false;
  bool backend_setup_completed_ = false;
  bool vtcm_backup_buffer_sharing_enabled_ = false;
  bool file_mapped_weights_enabled_ = false;

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  std::unique_ptr<FileMappingInterface> file_mapper_ = nullptr;
  // Notify params for file mapping must persist throughout lifetime of
  // QnnBackendManager for release of DMA data callback on destruction
  std::vector<std::unique_ptr<FileMappingCallbackInfo_t>> file_mapping_notify_params_;
#endif

  // NPU backend requires quantized model
  QnnBackendType qnn_backend_type_ = QnnBackendType::CPU;
  Qnn_ProfileHandle_t profile_backend_handle_ = nullptr;
  ContextPriority context_priority_;
  std::string sdk_build_version_ = "";
#ifdef _WIN32
  std::set<HMODULE> mod_handles_;
#endif
  const std::shared_ptr<QnnSerializerConfig> qnn_serializer_config_;
  uint32_t device_id_ = 0;
  QnnHtpDevice_Arch_t htp_arch_ = QNN_HTP_DEVICE_ARCH_NONE;
  uint32_t soc_model_ = QNN_SOC_MODEL_UNKNOWN;
  const std::vector<OpPackage> op_packages_;
  bool skip_qnn_version_check_ = false;

  // Mapping of thread id to on-run-start/end power configs
  std::mutex per_thread_power_configs_mutex_;
  std::unordered_map<std::thread::id, PerThreadHtpPowerConfigs_t> per_thread_power_configs_;

  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library_ = nullptr;
};

}  // namespace qnn
}  // namespace onnxruntime
