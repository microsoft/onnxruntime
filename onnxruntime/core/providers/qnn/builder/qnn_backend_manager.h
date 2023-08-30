// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <libloaderapi.h>
#else
#include <dlfcn.h>
#endif

#include "HTP/QnnHtpDevice.h"
#include "QnnLog.h"
#include "System/QnnSystemInterface.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/common/path_string.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class QnnModel;

class QnnBackendManager {
 public:
  QnnBackendManager(std::string backend_path,
                    ProfilingLevel profiling_level,
                    uint32_t rpc_control_latency,
                    HtpPerformanceMode htp_performance_mode)
      : backend_path_(backend_path),
        profiling_level_(profiling_level),
        rpc_control_latency_(rpc_control_latency),
        htp_performance_mode_(htp_performance_mode) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnBackendManager);

  ~QnnBackendManager();
  char* DlError() {
#ifdef _WIN32
    return "";
#else
    return ::dlerror();
#endif
  }

  Status LoadBackend();

  Status InitializeBackend();

  Status CreateDevice();

  Status ReleaseDevice();

  Status ShutdownBackend();

  Status InitializeProfiling();

  Status ReleaseProfilehandle();

  Status CreateContext();

  Status ReleaseContext();

  Status ResetContext() {
    ORT_RETURN_IF_ERROR(ReleaseContext());

    return CreateContext();
  }

  Status DumpQnnContext(const std::string& model_name, const std::string& graph_name);

  Status LoadCachedQnnContext(QnnModel& qnn_model);

  Status GetMetadataFromOrtContextFile();

  Status ValidateWithContextFile(const std::string& model_name, const std::string& graph_name);

  Status SetupBackend(const logging::Logger& logger, bool load_from_cached_context);

  Status SetHtpPowerConfig();

  const QNN_INTERFACE_VER_TYPE& GetQnnInterface() { return qnn_interface_; }

  const Qnn_ContextHandle_t& GetQnnContext() { return context_; }

  const Qnn_BackendHandle_t& GetQnnBackendHandle() { return backend_handle_; }

  const Qnn_ProfileHandle_t& GetQnnProfileHandle() { return profile_backend_handle_; }

  std::string GetBackendBuildId() {
    char* backend_build_id{nullptr};
    if (QNN_SUCCESS != qnn_interface_.backendGetBuildId((const char**)&backend_build_id)) {
      LOGS(*logger_, ERROR) << "Unable to get build Id from the backend.";
    }
    return (backend_build_id == nullptr ? std::string("") : std::string(backend_build_id));
  }

  void SetLogger(const logging::Logger* logger) {
    if (logger_ == nullptr) {
      logger_ = logger;
      InitializeQnnLog();
    }
  }

  void InitializeQnnLog();

  // Terminate logging in the backend
  Status TerminateQnnLog() {
    if (logger_ == nullptr) {
      return Status::OK();
    }

    if (nullptr != qnn_interface_.logFree && nullptr != log_handle_) {
      ORT_RETURN_IF(QNN_SUCCESS != qnn_interface_.logFree(log_handle_),
                    "Unable to terminate logging in the backend.");
    }

    return Status::OK();
  }

  void ReleaseResources();

  void Split(std::vector<std::string>& split_string, const std::string& tokenized_string, const char separator);

  Status ExtractBackendProfilingInfo();
  Status ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id);
  Status ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id);

  // NPU backend requires quantized model
  bool IsNpuBackend() { return is_npu_backend_; }

  bool IsContextCacheFileExists(const std::string& customer_context_cache_path,
                                const std::string& model_description,
                                const onnxruntime::PathString& model_pathstring);

 private:
  void* LoadLib(const char* file_name, int flags, std::string& error_msg);

  Status LoadQnnSystemLib();

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
  Status GetQnnInterfaceProviders(const char* lib_path,
                                  const char* interface_provider_name,
                                  void** backend_lib_handle,
                                  T*** interface_providers,
                                  uint32_t& num_providers);

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

 private:
  const std::string backend_path_;
  const logging::Logger* logger_ = nullptr;
  QNN_INTERFACE_VER_TYPE qnn_interface_ = QNN_INTERFACE_VER_TYPE_INIT;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_sys_interface_ = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
  void* backend_lib_handle_ = nullptr;
  void* system_lib_handle_ = nullptr;
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  QnnBackend_Config_t** backend_config_ = nullptr;
  Qnn_LogHandle_t log_handle_ = nullptr;
  Qnn_DeviceHandle_t device_handle_ = nullptr;
  Qnn_ContextHandle_t context_ = nullptr;
  QnnContext_Config_t** context_config_ = nullptr;
  ProfilingLevel profiling_level_;
  bool backend_initialized_ = false;
  bool device_created_ = false;
  bool context_created_ = false;
  bool backend_setup_completed_ = false;
  // NPU backend requires quantized model
  bool is_npu_backend_ = false;
  Qnn_ProfileHandle_t profile_backend_handle_ = nullptr;
  std::vector<std::string> op_package_paths_;
  uint32_t rpc_control_latency_ = 0;
  HtpPerformanceMode htp_performance_mode_;
  std::string model_name_from_ctx_cache_ = "";
  std::string graph_name_from_ctx_cache_ = "";
  std::string model_description_from_ctx_cache_ = "";
  std::string model_description_ = "";
  std::string context_cache_path_ = "";
  bool ctx_file_exists_ = false;
  bool ctx_metadata_tried_ = false;
  bool ort_generated_ctx_cache_ = false;
  bool get_capability_round_2_ = false;
  uint16_t ort_ctx_metadata_length_ = 0;
#ifdef _WIN32
  std::set<HMODULE> mod_handles_;
#endif
};

}  // namespace qnn
}  // namespace onnxruntime
