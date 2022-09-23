// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _WIN32
#include <windows.h>
#define LIBTYPE HINSTANCE
#define OPENLIB(libname) LoadLibrary(libname)
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#define DLERROR() ""
#else
#include <dlfcn.h>
#define LIBTYPE void*
#define OPENLIB(libname) dlopen((libname), RTLD_NOW | RTLD_GLOBAL)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#define DLERROR() dlerror()
#endif

#include "QnnLog.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class QnnBackendManager {
 public:
  QnnBackendManager(std::string backend_path,
                    ProfilingLevel profiling_level,
                    bool is_dsp_backend,
                    uint32_t rpc_control_latency)
      : backend_path_(backend_path),
        logger_(nullptr),
        qnn_interface_(QNN_INTERFACE_VER_TYPE_INIT),
        backend_handle_(nullptr),
        backend_config_(nullptr),
        context_(nullptr),
        context_config_(nullptr),
        profiling_level_(profiling_level),
        backend_initialized_(false),
        context_created_(false),
        backend_setup_completed_(false),
        is_dsp_backend_(is_dsp_backend),
        profile_backend_handle_(nullptr),
        rpc_control_latency_(rpc_control_latency) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnBackendManager);

  ~QnnBackendManager();

  Status LoadBackend();

  Status InitializeBackend();

  Status ShutdownBackend();

  Status InitializeProfiling();

  Status ReleaseProfilehandle();

  Status CreateContext();

  Status ReleaseContext();

  Status ResetContext() {
    ORT_RETURN_IF_ERROR(ReleaseContext());

    return CreateContext();
  }

  Status SetupBackend(const logging::Logger* logger);

  Status SetDspPowerConfig();

  QNN_INTERFACE_VER_TYPE GetQnnInterface() { return qnn_interface_; }

  Qnn_ContextHandle_t GetQnnContext() { return context_; }

  void* GetQnnBackendHandle() { return backend_handle_; }

  Qnn_ProfileHandle_t GetQnnProfileHandle() { return profile_backend_handle_; }

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

  void InitializeQnnLog() {
    const std::map<logging::Severity, QnnLog_Level_t> ort_log_level_to_qnn_log_level = {
        {logging::Severity::kVERBOSE, QNN_LOG_LEVEL_DEBUG},
        {logging::Severity::kINFO, QNN_LOG_LEVEL_INFO},
        {logging::Severity::kWARNING, QNN_LOG_LEVEL_WARN},
        {logging::Severity::kERROR, QNN_LOG_LEVEL_ERROR},
        {logging::Severity::kFATAL, QNN_LOG_LEVEL_ERROR}};

    QnnLog_Level_t qnn_log_level = QNN_LOG_LEVEL_WARN;
    auto ort_log_level = logger_->GetSeverity();
    auto pos = ort_log_level_to_qnn_log_level.find(ort_log_level);
    if (pos != ort_log_level_to_qnn_log_level.end()) {
      qnn_log_level = pos->second;
    }

    if (QNN_SUCCESS != qnn_interface_.logInitialize(QnnLogStdoutCallback, qnn_log_level)) {
      LOGS(*logger_, WARNING) << "Unable to initialize logging in the QNN backend.";
    }
  }

  // Terminate logging in the backend
  void TerminateQnnLog() {
    LOGS_DEFAULT(VERBOSE) << "Terminate Qnn log.";
    if (logger_ == nullptr) {
      return;
    }

    if (QNN_SUCCESS != qnn_interface_.logTerminate()) {
      LOGS_DEFAULT(WARNING) << "Unable to terminate logging in the backend.";
    }
    LOGS_DEFAULT(VERBOSE) << "Terminate Qnn log succeed.";
  }

  void ReleaseResources();

  void Split(std::vector<std::string>& split_string, const std::string& tokenized_string, const char separator);

  Status ExtractBackendProfilingInfo();
  Status ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id);
  Status ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id);

 private:
  const std::string backend_path_;
  const logging::Logger* logger_;
  QNN_INTERFACE_VER_TYPE qnn_interface_;
  LIBTYPE backend_handle_;
  QnnBackend_Config_t** backend_config_;
  Qnn_ContextHandle_t context_;
  QnnContext_Config_t** context_config_;
  ProfilingLevel profiling_level_;
  bool backend_initialized_;
  bool context_created_;
  bool backend_setup_completed_;
  bool is_dsp_backend_;
  Qnn_ProfileHandle_t profile_backend_handle_;
  std::vector<std::string> op_package_paths_;
  uint32_t rpc_control_latency_;
};

}  // namespace qnn
}  // namespace onnxruntime
