//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_backend_manager.h"
#include "qnn_model.h"
#include <filesystem>
#include <fstream>
#include <string>
#include "QnnOpDef.h"
#include "CPU/QnnCpuCommon.h"
#include "GPU/QnnGpuCommon.h"
#include "DSP/QnnDspCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpSystemContext.h"
#include "IR/QnnIrCommon.h"
#include "IR/QnnIrGraph.h"
#include "Saver/QnnSaver.h"
#include "Saver/QnnSaverCommon.h"
#include <gsl/gsl>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_telemetry.h"
#include "core/providers/qnn/shared_context.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/qnn_configs_helper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#endif

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

// Ensure that we have a recent enough version of QNN
static_assert(QNN_API_VERSION_MAJOR > 2 ||
                  (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 29),
              "Minimum required QAIRT SDK version is 2.39.0");

namespace onnxruntime {
namespace qnn {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                          uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList,
                                                                uint32_t* numProviders);

static Qnn_Version_t GetQnnInterfaceApiVersion(const QnnInterface_t* qnn_interface) {
  return qnn_interface->apiVersion.coreApiVersion;
}

static Qnn_Version_t GetQnnInterfaceApiVersion(const QnnSystemInterface_t* qnn_interface) {
  return qnn_interface->systemApiVersion;
}

static const char* DlError() {
#ifdef _WIN32
  return "";
#else
  return ::dlerror();
#endif
}

// Workaround for a missing comma in QNN_IR_GRAPH_CUSTOM_CONFIG_INIT.
static QnnIrGraph_CustomConfig_t EmptyIrGraphConfig() {
  return {
      QNN_IR_GRAPH_CONFIG_OPTION_SERIALIZATION, {QNN_IR_GRAPH_SERIALIZATION_TYPE_FLAT_BUFFER, ""}};
}

class QnnIrConfig : public QnnSerializerConfig {
 public:
  QnnIrConfig(std::string backend_path, std::string dlc_dir)
      : QnnSerializerConfig(std::move(backend_path)), dlc_dir_(std::move(dlc_dir)), configs_builder_(MakeConfigsBuilder()) {
  }

  const QnnGraph_Config_t** Configure() override {
    auto configs_builder = MakeConfigsBuilder();

    std::filesystem::path dlc_path = (dlc_dir_ / (GetGraphName() + ".dlc"));
    std::string dlc_path_str = dlc_path.string();
    gsl::not_null<QnnIrGraph_CustomConfig_t*> dlc_path_config = configs_builder.PushCustomConfig();
    dlc_path_config->option = QNN_IR_GRAPH_CONFIG_OPTION_SERIALIZATION;
    dlc_path_config->serializationOption.serializationType = QNN_IR_GRAPH_SERIALIZATION_TYPE_FLAT_BUFFER;
    dlc_path_config->serializationOption.outputPath = dlc_path_str.c_str();

    gsl::not_null<QnnGraph_Config_t*> dlc_path_custom_config = configs_builder.PushConfig();
    dlc_path_custom_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    dlc_path_custom_config->customConfig = dlc_path_config;

    std::filesystem::create_directories(dlc_path);

    // Keep the pointer to dlc_path_str's null-terminated string alive.
    std::swap(dlc_path_str, dlc_path_str_);

    std::swap(configs_builder, configs_builder_);
    return configs_builder_.GetQnnConfigs();
  }

  bool SupportsArbitraryGraphConfigs() const override {
    return false;
  }

 private:
  static QnnConfigsBuilder<QnnGraph_Config_t, QnnIrGraph_CustomConfig_t> MakeConfigsBuilder() {
    return QnnConfigsBuilder<QnnGraph_Config_t, QnnIrGraph_CustomConfig_t>(QNN_GRAPH_CONFIG_INIT, EmptyIrGraphConfig());
  }

  std::filesystem::path dlc_dir_;
  std::string dlc_path_str_;
  QnnConfigsBuilder<QnnGraph_Config_t, QnnIrGraph_CustomConfig_t> configs_builder_;
};

class QnnSaverConfig : public QnnSerializerConfig {
 public:
  QnnSaverConfig(std::string backend_path) : QnnSerializerConfig(std::move(backend_path)) {}

  const QnnGraph_Config_t** Configure() override {
    return nullptr;
  }

  bool SupportsArbitraryGraphConfigs() const override {
    return true;
  }
};

QnnSerializerConfig::~QnnSerializerConfig() = default;

QnnSerializerConfig::QnnSerializerConfig(std::string backend_path)
    : backend_path_(std::move(backend_path)) {}

std::unique_ptr<QnnSerializerConfig> QnnSerializerConfig::CreateIr(std::string backend_path, std::string dlc_dir) {
  return std::make_unique<QnnIrConfig>(std::move(backend_path), std::move(dlc_dir));
}

std::unique_ptr<QnnSerializerConfig> QnnSerializerConfig::CreateSaver(std::string backend_path) {
  return std::make_unique<QnnSaverConfig>(std::move(backend_path));
}

const std::string& QnnSerializerConfig::GetBackendPath() const {
  return backend_path_;
}

const std::string& QnnSerializerConfig::GetGraphName() const {
  return graph_name_;
}

void QnnSerializerConfig::SetGraphName(std::string graph_name) {
  graph_name_ = std::move(graph_name);
}

Status ReadBinaryFromFile(const std::string& file_path, uint8_t* buffer, size_t buffer_size) {
  ORT_RETURN_IF(nullptr == buffer, "Binary buffer is nullptr");
  std::ifstream in(file_path, std::ifstream::binary);
  ORT_RETURN_IF(!in, "Failed to open input file: ", file_path.c_str());
  ORT_RETURN_IF(!in.read(reinterpret_cast<char*>(buffer), buffer_size), "Failed to read the contents of: ", file_path.c_str());
  return Status::OK();
}

Status QnnBackendManager::ParseLoraConfig(std::string lora_config_path) {
  LOGS_DEFAULT(INFO) << "Acquiring the QnnInterface " << lora_config_path;

  // QNN Lora Config file format should be a single line, with the graph name first,
  // followed by the qnn lora context binary path, separated by a semicolon (;)
  // Example: <graph_name>;<binary_path>
  LOGS_DEFAULT(INFO) << "Loading Lora Config " << lora_config_path;
  std::ifstream file(lora_config_path);
  std::string line;

  if (file.is_open()) {
    if (std::getline(file, line)) {
      std::istringstream ss(line);
      std::string graph_name;
      std::string lora_adapter_bin_path;

      if (std::getline(ss, graph_name, ';') && std::getline(ss, lora_adapter_bin_path)) {
        size_t buffer_size = std::filesystem::file_size(lora_adapter_bin_path.c_str());

        ORT_RETURN_IF(0 == buffer_size, "Received path to an empty file. Nothing to deserialize.");
        std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(buffer_size);
        void* voidBufferPtr = static_cast<void*>(buffer.get());
        QnnContext_Buffer_t contextBuffer{QNN_CONTEXT_BUFFER_VERSION_1,
                                          {QNN_CONTEXTMEMTYPE_RAW, {{voidBufferPtr, buffer_size}}}};

        auto status = ReadBinaryFromFile(lora_adapter_bin_path,
                                         reinterpret_cast<uint8_t*>(buffer.get()),
                                         buffer_size);

        ORT_RETURN_IF(status != Status::OK(), "Failed to read binary data.");
        Qnn_GraphHandle_t graph;
        bool graph_retrieve_success = false;
        for (size_t cIdx = 0; cIdx < contexts_.size(); cIdx++) {
          auto graph_retrieve_rt = qnn_interface_.graphRetrieve(contexts_[cIdx], graph_name.c_str(), &graph);
          if (QNN_SUCCESS != graph_retrieve_rt) {
            continue;
          }

          graph_retrieve_success = true;

          auto context_apply_binary_section_rt = qnn_interface_.contextApplyBinarySection(
              contexts_[cIdx], graph, QNN_CONTEXT_SECTION_UPDATABLE, &contextBuffer, profile_backend_handle_, nullptr);
          ORT_RETURN_IF(QNN_SUCCESS != context_apply_binary_section_rt, "Failed to apply binary section.");
          break;
        }
        ORT_RETURN_IF_NOT(graph_retrieve_success, "Failed to retrieve graph: ", graph_name, " and apply binary section.");
      }
    }
    file.close();
  } else {
    LOGS_DEFAULT(ERROR) << "Unable to load Lora Config " << lora_config_path;
  }

  return Status::OK();
}

template <typename F, class T>
Status QnnBackendManager::GetQnnInterfaceProvider(const char* lib_path,
                                                  const char* interface_provider_name,
                                                  void** backend_lib_handle,
                                                  Qnn_Version_t req_version,
                                                  T** interface_provider) {
  std::string error_msg;
  *backend_lib_handle = LoadLib(lib_path,
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_GLOBAL),
                                error_msg);
  ORT_RETURN_IF(nullptr == *backend_lib_handle, "Unable to load backend, error: ", error_msg, " ", DlError());

  // Get QNN Interface providers
  F GetInterfaceProviders{nullptr};
  GetInterfaceProviders = ResolveSymbol<F>(*backend_lib_handle, interface_provider_name, *logger_);
  ORT_RETURN_IF(nullptr == GetInterfaceProviders, "Failed to get QNN providers!");

  T** interface_providers{nullptr};
  uint32_t num_providers{0};

  auto result = GetInterfaceProviders((const T***)&interface_providers, &num_providers);
  ORT_RETURN_IF((QNN_SUCCESS != result || nullptr == *interface_providers || 0 == num_providers),
                "Failed to get QNN providers.");

  if (skip_qnn_version_check_) {
    // When skipping version check, use the first available provider.
    *interface_provider = interface_providers[0];
    return Status::OK();
  }

  bool found_valid_interface{false};
  for (size_t pIdx = 0; pIdx < num_providers; pIdx++) {
    Qnn_Version_t interface_version = GetQnnInterfaceApiVersion(interface_providers[pIdx]);

    LOGS_DEFAULT(VERBOSE) << lib_path << " interface version: " << interface_version.major << "."
                          << interface_version.minor << "." << interface_version.patch;

    // Check the interface's API version against the required version.
    // Major versions must match. The interface's minor version must be greater OR equal with a suitable patch version.
    if (interface_version.major == req_version.major) {
      bool minor_and_patch_version_ok = (interface_version.minor > req_version.minor) ||
                                        (interface_version.minor == req_version.minor &&
                                         interface_version.patch >= req_version.patch);
      if (minor_and_patch_version_ok) {
        found_valid_interface = true;
        *interface_provider = interface_providers[pIdx];
        break;
      }
    }
  }

  ORT_RETURN_IF_NOT(found_valid_interface, "Unable to find a valid interface for ", lib_path);

  return Status::OK();
}

void QnnBackendManager::SetQnnBackendType(uint32_t backend_id) {
  switch (backend_id) {
    case QNN_BACKEND_ID_CPU:
      qnn_backend_type_ = QnnBackendType::CPU;
      break;
    case QNN_BACKEND_ID_GPU:
      qnn_backend_type_ = QnnBackendType::GPU;
      break;
    case QNN_BACKEND_ID_DSP:
      qnn_backend_type_ = QnnBackendType::DSP;
      break;
    case QNN_BACKEND_ID_HTP:
      qnn_backend_type_ = QnnBackendType::HTP;
      break;
    case QNN_BACKEND_ID_IR:
    case QNN_BACKEND_ID_SAVER:
      qnn_backend_type_ = QnnBackendType::SERIALIZER;
      break;
    default:
      qnn_backend_type_ = QnnBackendType::CPU;
      break;
  }
}

Status QnnBackendManager::LoadBackend() {
#if defined(__aarch64__) && defined(__linux__)
  // QNN requires ADSP_LIBRARY_PATH to be set in order to find skel libs on Linux
  static std::once_flag set_adsp_path_once;

  std::call_once(set_adsp_path_once, []() {
    constexpr std::string_view kAdspLibraryPathEnvVar{"ADSP_LIBRARY_PATH"};
    const char* existing_path = getenv(kAdspLibraryPathEnvVar.data());
    if (existing_path != nullptr) {
      LOGS_DEFAULT(WARNING) << "Using existing ADSP_LIBRARY_PATH setting of " << existing_path
                            << ", which may cause the HTP backend to fail.";
      return;
    }

    std::filesystem::path qnn_lib_path(GetDefaultEnv().GetRuntimePath());
    LOGS_DEFAULT(INFO) << "Setting " << kAdspLibraryPathEnvVar << " = " << qnn_lib_path;
    setenv(kAdspLibraryPathEnvVar.data(), qnn_lib_path.c_str(), 1);
  });
#endif

  QnnInterface_t* backend_interface_provider{nullptr};
  auto rt = GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t,
                                    QnnInterface_t>(backend_path_.c_str(),
                                                    "QnnInterface_getProviders",
                                                    &backend_lib_handle_,
                                                    {QNN_API_VERSION_MAJOR,
                                                     QNN_API_VERSION_MINOR,
                                                     QNN_API_VERSION_PATCH},
                                                    &backend_interface_provider);
  ORT_RETURN_IF_ERROR(rt);
  qnn_interface_ = backend_interface_provider->QNN_INTERFACE_VER_NAME;
  auto backend_id = backend_interface_provider->backendId;
  SetQnnBackendType(backend_id);

  Qnn_Version_t backend_interface_version = GetQnnInterfaceApiVersion(backend_interface_provider);
  LOGS_DEFAULT(INFO) << "Found valid interface, version: " << backend_interface_version.major
                     << "." << backend_interface_version.minor << "." << backend_interface_version.patch
                     << " backend provider name: " << backend_interface_provider->providerName
                     << " backend id: " << backend_id;

  return Status::OK();
}

QnnSerializerConfig* QnnBackendManager::GetQnnSerializerConfig() {
  return qnn_serializer_config_.get();
}

// Loads the intended backend (e.g., HTP, CPU, etc) to get its type, and then
// sets QnnSaver or QnnIr as the active backend. QNN op builders will still see the intended backend
// (e.g., HTP) as the backend type to ensure they emit the expected QNN API calls. Note, however, that
// calls to QnnBackend_validateOpConfig will be to the saver backend, not the "intended" one.
//
// QnnSaver and QnnIr are "debugging" backends that serializes all QNN API calls (and weights) into
// local files: Saver dumps to C++ sources and Ir to .dlc archives.
// This information can be used to debug issues by replaying QNN API calls with another backend.
Status QnnBackendManager::LoadQnnSerializerBackend() {
  void* backend_lib_handle = nullptr;

  // Helper that unloads the intended backend library handle when the `unload_backend_lib` variable
  // goes out of scope. Similar to `defer` in other languages.
  auto unload_backend_lib = gsl::finally([&] {
    if (backend_lib_handle != nullptr) {
      auto result = UnloadLib(backend_lib_handle);
      if (Status::OK() != result) {
        ORT_THROW("Failed to unload backend library.");
      }
    }
  });

  // Load the intended backend (e.g., HTP, CPU) to ensure it is valid and to get its type.
  QnnInterface_t* backend_interface_provider{nullptr};
  auto rt = GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t,
                                    QnnInterface_t>(backend_path_.c_str(),
                                                    "QnnInterface_getProviders",
                                                    &backend_lib_handle,
                                                    {QNN_API_VERSION_MAJOR,
                                                     QNN_API_VERSION_MINOR,
                                                     QNN_API_VERSION_PATCH},
                                                    &backend_interface_provider);
  ORT_RETURN_IF_ERROR(rt);

  // Set the "intended" backend type so that QNN builders still make the expected QNN API calls.
  auto backend_id = backend_interface_provider->backendId;
  SetQnnBackendType(backend_id);

  // Load the serializer backend and set it as the activate backend.
  QnnInterface_t* serializer_interface_provider{nullptr};
  auto saver_rt = GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t,
                                          QnnInterface_t>(qnn_serializer_config_->GetBackendPath().c_str(),
                                                          "QnnInterface_getProviders",
                                                          &backend_lib_handle_,  // NOTE: QnnSaver/Ir library handle is set
                                                          {QNN_API_VERSION_MAJOR,
                                                           QNN_API_VERSION_MINOR,
                                                           QNN_API_VERSION_PATCH},
                                                          &serializer_interface_provider);
  ORT_RETURN_IF_ERROR(saver_rt);
  qnn_interface_ = serializer_interface_provider->QNN_INTERFACE_VER_NAME;  // NOTE: QnnSaver/Ir will provide the interfaces

  Qnn_Version_t backend_interface_version = GetQnnInterfaceApiVersion(backend_interface_provider);
  Qnn_Version_t serializer_interface_version = GetQnnInterfaceApiVersion(serializer_interface_provider);

  LOGS_DEFAULT(INFO) << "Using QnnSaver/Ir version: " << serializer_interface_version.major << "."
                     << serializer_interface_version.minor << "." << serializer_interface_version.patch
                     << " provider name : " << serializer_interface_provider->providerName;

  LOGS_DEFAULT(INFO) << "Intended backend provider name: " << backend_interface_provider->providerName
                     << " backend id: " << backend_id
                     << " interface version: " << backend_interface_version.major
                     << "." << backend_interface_version.minor << "." << backend_interface_version.patch;

  return Status::OK();
}

Status QnnBackendManager::LoadQnnSystemLib() {
  if (!system_lib_loaded_) {
#ifdef _WIN32
    std::string system_lib_file = "QnnSystem.dll";
#else
    std::string system_lib_file = "libQnnSystem.so";
#endif  // #ifdef _WIN32
    LOGS_DEFAULT(INFO) << "Loading QnnSystem lib";
    std::filesystem::path lib_file_path(backend_path_.c_str());
    std::string sys_file_path(lib_file_path.remove_filename().string() + system_lib_file);
    QnnSystemInterface_t* system_interface_provider{nullptr};
    auto rt = GetQnnInterfaceProvider<QnnSystemInterfaceGetProvidersFn_t,
                                      QnnSystemInterface_t>(sys_file_path.c_str(),
                                                            "QnnSystemInterface_getProviders",
                                                            &system_lib_handle_,
                                                            {QNN_SYSTEM_API_VERSION_MAJOR,
                                                             QNN_SYSTEM_API_VERSION_MINOR,
                                                             QNN_SYSTEM_API_VERSION_PATCH},
                                                            &system_interface_provider);
    ORT_RETURN_IF_ERROR(rt);
    Qnn_Version_t system_interface_version = GetQnnInterfaceApiVersion(system_interface_provider);
    qnn_sys_interface_ = system_interface_provider->QNN_SYSTEM_INTERFACE_VER_NAME;

    LOGS_DEFAULT(INFO) << "Found valid system interface, version: " << system_interface_version.major
                       << "." << system_interface_version.minor
                       << " backend provider name: " << system_interface_provider->providerName;

    system_lib_loaded_ = true;
  }
  return Status::OK();
}

void QnnLogging(const char* format,
                QnnLog_Level_t level,
                uint64_t timestamp,
                va_list argument_parameter) {
  ORT_UNUSED_PARAMETER(timestamp);

  if (!::onnxruntime::logging::LoggingManager::HasDefaultLogger()) {
    // QNN may call this logging callback at any point, which means that we need to explicitly check
    // that the default logger has been initialized before trying to use it (otherwise get segfault).
    return;
  }

  const auto& logger = ::onnxruntime::logging::LoggingManager::DefaultLogger();
  // Map QNN log level to ORT severity
  logging::Severity severity = QnnBackendManager::MapQNNLogLevelToOrtSeverity(level);
  const auto data_type = ::onnxruntime::logging::DataType::SYSTEM;

  if (logger.OutputIsEnabled(severity, data_type)) {
    auto log_capture = Factory<logging::Capture>::Create(logger,
                                                         severity,
                                                         logging::Category::onnxruntime,
                                                         data_type,
                                                         ORT_WHERE);
    log_capture->ProcessPrintf(format, argument_parameter);
  }
}

Status QnnBackendManager::InitializeQnnLog(const logging::Logger& logger) {
  logger_ = &logger;

  // Set Qnn log level align with Ort log level
  auto ort_log_level = logger_->GetSeverity();
  QnnLog_Level_t qnn_log_level = MapOrtSeverityToQNNLogLevel(ort_log_level);
  LOGS(*logger_, VERBOSE) << "Set Qnn log level: " << qnn_log_level;

  // NOTE: Even if logCreate() fails and QNN does not return a valid log_handle_, QNN may still
  // call the QnnLogging() callback. So, we have to make sure that QnnLogging() can handle calls
  // in which ORT logging is not available.
  Qnn_ErrorHandle_t result = qnn_interface_.logCreate(QnnLogging, qnn_log_level, &log_handle_);

  if (result != QNN_SUCCESS) {
    switch (result) {
      case QNN_COMMON_ERROR_NOT_SUPPORTED:
        LOGS(*logger_, ERROR) << "Logging not supported in the QNN backend.";
        break;
      case QNN_LOG_ERROR_INVALID_ARGUMENT:
        LOGS(*logger_, ERROR) << "Invalid argument provided to QnnLog_create.";
        break;
      case QNN_LOG_ERROR_MEM_ALLOC:
        LOGS(*logger_, ERROR) << "Memory allocation error during QNN logging initialization.";
        break;
      case QNN_LOG_ERROR_INITIALIZATION:
        LOGS(*logger_, ERROR) << "Initialization of logging failed in the QNN backend.";
        break;
      default:
        LOGS(*logger_, WARNING) << "Unknown error occurred while initializing logging in the QNN backend.";
        break;
    }
  }

  ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != result, "Failed to initialize logging in the QNN backend. Error: ", QnnErrorHandleToString(result));
  return Status::OK();
}

QnnLog_Level_t QnnBackendManager::MapOrtSeverityToQNNLogLevel(logging::Severity ort_log_level) {
  // Map ORT log severity to Qnn log level
  switch (ort_log_level) {
    case logging::Severity::kVERBOSE: {
      switch ((GetQnnBackendType())) {
        case QnnBackendType::GPU:
          // Currently GPU needs this log level to work.
          // This switch will be removed once this is resolved.
          return QNN_LOG_LEVEL_DEBUG;
        default:
          return QNN_LOG_LEVEL_VERBOSE;
      }
    }
    case logging::Severity::kINFO:
      return QNN_LOG_LEVEL_INFO;
    case logging::Severity::kWARNING:
      return QNN_LOG_LEVEL_WARN;
    case logging::Severity::kERROR:
    case logging::Severity::kFATAL:
    default:
      return QNN_LOG_LEVEL_ERROR;
  }
}

/* static */ logging::Severity QnnBackendManager::MapQNNLogLevelToOrtSeverity(QnnLog_Level_t qnn_log_level) {
  // Map QNN log level to ORT log severity
  switch (qnn_log_level) {
    case QNN_LOG_LEVEL_VERBOSE:
    case QNN_LOG_LEVEL_DEBUG:
      return logging::Severity::kVERBOSE;
    case QNN_LOG_LEVEL_INFO:
      return logging::Severity::kINFO;
    case QNN_LOG_LEVEL_WARN:
      return logging::Severity::kWARNING;
    case QNN_LOG_LEVEL_ERROR:
    default:
      return logging::Severity::kERROR;
  }
}

Status QnnBackendManager::ResetQnnLogLevel(std::optional<logging::Severity> ort_log_level) {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (!backend_setup_completed_ || logger_ == nullptr) {
    return Status::OK();
  }
  ORT_RETURN_IF(nullptr == log_handle_, "Unable to update QNN Log Level. Invalid QNN log handle.");

  logging::Severity actual_log_level = ort_log_level.has_value() ? *ort_log_level : logger_->GetSeverity();
  QnnLog_Level_t qnn_log_level = MapOrtSeverityToQNNLogLevel(actual_log_level);

  LOGS(*logger_, INFO) << "Updating Qnn log level to: " << qnn_log_level;

  // Use the QnnLog_setLogLevel API to set the new log level
  Qnn_ErrorHandle_t result = qnn_interface_.logSetLogLevel(log_handle_, qnn_log_level);
  if (QNN_SUCCESS != result) {
    if (result == QNN_LOG_ERROR_INVALID_ARGUMENT) {
      LOGS(*logger_, ERROR) << "Invalid log level argument provided to QnnLog_setLogLevel.";
    } else if (result == QNN_LOG_ERROR_INVALID_HANDLE) {
      LOGS(*logger_, ERROR) << "Invalid log handle provided to QnnLog_setLogLevel.";
    }
  }
  ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != result,
                "Failed to set log level in Qnn backend. Error: ", QnnErrorHandleToString(result));
  return Status::OK();
}

Status QnnBackendManager::InitializeBackend() {
  if (true == backend_initialized_) {
    LOGS_DEFAULT(INFO) << "Backend initialized already.";
    return Status::OK();
  }

  Qnn_ErrorHandle_t result = qnn_interface_.backendCreate(log_handle_, (const QnnBackend_Config_t**)backend_config_, &backend_handle_);
  ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != result, "Failed to initialize backend. Error: ", QnnErrorHandleToString(result));

  backend_initialized_ = true;
  return Status::OK();
}

Status QnnBackendManager::ShutdownBackend() {
  if (false == backend_initialized_) {
    return Status::OK();
  }

  if (nullptr != qnn_interface_.backendFree) {
    ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != qnn_interface_.backendFree(backend_handle_),
                  "Failed to shutdown backend!");
  }

  backend_initialized_ = false;

  return Status::OK();
}

bool QnnBackendManager::IsDevicePropertySupported() {
  if (nullptr != qnn_interface_.propertyHasCapability) {
    auto rt = qnn_interface_.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == rt || QNN_PROPERTY_ERROR_UNKNOWN_KEY == rt) {
      LOGS_DEFAULT(INFO) << "Device property not supported or unknown to backend.";
      return false;
    }
  }

  return true;
}

Status QnnBackendManager::CreateDevice() {
  if (true == device_created_) {
    LOGS_DEFAULT(INFO) << "Device initialized already.";
    return Status::OK();
  }

  // Create device if its property supported
  if (!IsDevicePropertySupported()) {
    LOGS_DEFAULT(INFO) << "Skip to create device.";
    return Status::OK();
  }

  qnn::QnnConfigsBuilder<QnnDevice_Config_t, QnnHtpDevice_CustomConfig_t> device_configs_builder(QNN_DEVICE_CONFIG_INIT,
                                                                                                 {});
  if (qnn_backend_type_ == QnnBackendType::HTP) {
    // Set SoC Model. The *enum* Qnn_SocModel_t is deprecated and will not be updated in the future. Therefore,
    // must use the latest SDK documentation to get the SoC model of the latest HW.
    if (soc_model_ != QNN_SOC_MODEL_UNKNOWN) {
      gsl::not_null<QnnHtpDevice_CustomConfig_t*> custom_config = device_configs_builder.PushCustomConfig();
      custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
      custom_config->socModel = soc_model_;

      gsl::not_null<QnnDevice_Config_t*> device_config = device_configs_builder.PushConfig();
      device_config->option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      device_config->customConfig = custom_config;
    }

    // Set the minimum HTP architecture. The driver will use ops that are compatible with this minimum architecture.
    if (htp_arch_ != QNN_HTP_DEVICE_ARCH_NONE) {
      gsl::not_null<QnnHtpDevice_CustomConfig_t*> custom_config = device_configs_builder.PushCustomConfig();
      custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
      custom_config->arch.arch = htp_arch_;
      custom_config->arch.deviceId = device_id_;

      gsl::not_null<QnnDevice_Config_t*> device_config = device_configs_builder.PushConfig();
      device_config->option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      device_config->customConfig = custom_config;
    }
  }

  LOGS_DEFAULT(INFO) << "Create device.";
  if (nullptr != qnn_interface_.deviceCreate) {
    Qnn_ErrorHandle_t result = qnn_interface_.deviceCreate(log_handle_, device_configs_builder.GetQnnConfigs(), &device_handle_);
    if (QNN_SUCCESS != result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create device. Error: ", QnnErrorHandleToString(result));
    }
  }
  device_created_ = true;

  return Status::OK();
}

Status QnnBackendManager::ReleaseDevice() {
  if (false == device_created_) {
    return Status::OK();
  }

  if (nullptr != qnn_interface_.deviceFree) {
    Qnn_ErrorHandle_t result = qnn_interface_.deviceFree(device_handle_);
    if (QNN_SUCCESS != result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to release device. Error: ", QnnErrorHandleToString(result));
    }
  }

  device_created_ = false;

  return Status::OK();
}

Status QnnBackendManager::InitializeProfiling() {
  profiling_level_merge_ = profiling_level_;
  // use profiling level from ETW if ETW is enabled
  if (profiling_level_etw_ != ProfilingLevel::INVALID) {
    profiling_level_merge_ = profiling_level_etw_;
  }

  if (ProfilingLevel::OFF == profiling_level_merge_ || ProfilingLevel::INVALID == profiling_level_merge_) {
    LOGS_DEFAULT(INFO) << "Profiling turned off.";
    return Status::OK();
  }

  bool enable_optrace = false;
  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  if (ProfilingLevel::BASIC == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
    LOGS_DEFAULT(VERBOSE) << "Profiling level set to basic.";
  } else if (ProfilingLevel::DETAILED == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    LOGS_DEFAULT(VERBOSE) << "Profiling level set to detailed.";
  } else if (ProfilingLevel::OPTRACE == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    enable_optrace = true;
    LOGS_DEFAULT(VERBOSE) << "Profiling level set to optrace.";
  }

  Qnn_ErrorHandle_t result = qnn_interface_.profileCreate(backend_handle_, qnn_profile_level, &profile_backend_handle_);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to create QNN profile! Error: ", QnnErrorHandleToString(result));

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  profiling_enabled_ = true;
  ORT_RETURN_IF_ERROR(LoadQnnSystemLib());

  if (enable_optrace) {
    QnnProfile_Config_t optrace_config = QNN_PROFILE_CONFIG_INIT;
    optrace_config.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
    optrace_config.enableOptrace = enable_optrace;

    const QnnProfile_Config_t* profile_configs[] = {&optrace_config, nullptr};
    result = qnn_interface_.profileSetConfig(profile_backend_handle_, profile_configs);

    ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to enable op trace! Error: ", QnnErrorHandleToString(result));
  }
#else
  if (enable_optrace) {
    LOGS_DEFAULT(WARNING) << "Profiling level set to optrace, but QNN SDK Version is older than 2.29.0. "
                          << "Profiling level will be set to detailed instead.";
  }
#endif

  return Status::OK();
}

Status QnnBackendManager::ReleaseProfilehandle() {
  // Free Profiling object if it was created
  if (nullptr != profile_backend_handle_) {
    ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != qnn_interface_.profileFree(profile_backend_handle_),
                  "Could not free backend profile handle!");
  }
  profile_backend_handle_ = nullptr;

  return Status::OK();
}

Status QnnBackendManager::SetProfilingLevelETW(ProfilingLevel profiling_level_etw_param) {
  if (profiling_level_etw_ != profiling_level_etw_param) {
    profiling_level_etw_ = profiling_level_etw_param;

    auto result = ReleaseProfilehandle();
    if (Status::OK() != result) {
      ORT_THROW("Failed to ReleaseProfilehandle for previous QNN profiling");
    }

    result = InitializeProfiling();
    if (Status::OK() != result) {
      ORT_THROW("Failed to Re-InitializeProfiling for QNN ETW profiling");
    }
  }
  return Status::OK();
}

Status SetQnnContextConfig(ContextPriority context_priority, QnnContext_Config_t& qnn_context_config) {
  qnn_context_config.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  switch (context_priority) {
    case ContextPriority::LOW: {
      qnn_context_config.priority = QNN_PRIORITY_LOW;
      break;
    }
    case ContextPriority::NORMAL: {
      qnn_context_config.priority = QNN_PRIORITY_NORMAL;
      break;
    }
    case ContextPriority::NORMAL_HIGH: {
      qnn_context_config.priority = QNN_PRIORITY_NORMAL_HIGH;
      break;
    }
    case ContextPriority::HIGH: {
      qnn_context_config.priority = QNN_PRIORITY_HIGH;
      break;
    }
    case ContextPriority::UNDEFINED: {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid Qnn context priority.");
    }
    default:
      qnn_context_config.priority = QNN_PRIORITY_NORMAL;
  }  // switch

  return Status::OK();
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
// Callback required for allocating file mapping resources
static Qnn_ErrorHandle_t MapDmaDataCallback(Qnn_ContextBinaryDataRequest_t request,
                                            Qnn_ContextBinaryDmaDataResponse_t* response, void* notify_param) {
  if (notify_param == nullptr) {
    LOGS_DEFAULT(ERROR) << "MapDmaDataCallback: notify_param is null";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  auto callback_info = reinterpret_cast<QnnBackendManager::FileMappingCallbackInfo_t*>(notify_param);

  if (callback_info->backend_manager == nullptr) {
    LOGS_DEFAULT(ERROR) << "MapDmaDataCallback: QnnBackendManager is null";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  return callback_info->backend_manager->MapDmaData(request, response,
                                                    callback_info->mapped_file_ptr,
                                                    callback_info->file_size);
}

Qnn_ErrorHandle_t QnnBackendManager::MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                                                Qnn_ContextBinaryDmaDataResponse_t* response,
                                                void* const mapped_base_ptr,
                                                const size_t file_size) {
  if (!file_mapped_weights_enabled_) {
    LOGS(*logger_, WARNING) << "Attempting to map DMA data but file mapping has been disabled, "
                            << "possibly due to an error in a previous request.";
    return QNN_CONTEXT_ERROR_ABORTED;
  }

  if (mapped_base_ptr == nullptr) {
    LOGS(*logger_, ERROR) << "Attempting to map DMA data for null memory mapped base pointer";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  LOGS(*logger_, INFO) << "Mapping DMA data for request: memory mapped base pointer("
                       << mapped_base_ptr << "), offset(" << request.offset
                       << "), size(" << request.size << "), total file size("
                       << file_size << ") isBackendMappingNeeded("
                       << request.isBackendMappingNeeded << ")";

  auto size = request.size;
  if (size == 0 || !request.isBackendMappingNeeded) {
    LOGS(*logger_, ERROR) << "Mapping request size must be > 0 with backend mapping required";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // offset & size are type uint64_t
  // Should never be an issue, but if this occurs then there is something inherently wrong with QNN
  if ((UINT64_MAX - request.offset) < size) {
    LOGS(*logger_, ERROR) << "Critical error in QNN: mapping request offset + size will overflow 64 bits";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // file_size will be promoted to 64 bits on 32-bit systems
  if ((request.offset + size) > file_size) {
    LOGS(*logger_, ERROR) << "Requested offset and size includes memory outside of mapped file";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  void* unaligned_data_ptr = static_cast<char*>(mapped_base_ptr) + request.offset;
  rpcmem_library_->Api().register_buf(unaligned_data_ptr, size, NULL,
                                      rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

  auto fd = rpcmem_library_->Api().to_fd(unaligned_data_ptr);
  if (fd == -1) {
    LOGS(*logger_, ERROR) << "Failed to register DMA data mapping to RPCMEM";
    return QNN_COMMON_ERROR_SYSTEM;
  }

  LOGS(*logger_, INFO) << "Created DMA data mapping with address: " << unaligned_data_ptr;

  response->dmaBuffer.fd = fd;
  response->dmaBuffer.data = unaligned_data_ptr;
  response->dataStartOffset = 0;
  response->alignedSize = size;

  return QNN_SUCCESS;
}

// Callback required for releasing file mapping resources
static Qnn_ErrorHandle_t ReleaseDmaDataCallback(Qnn_ContextBinaryDmaDataMem_t data_mem, void* notify_param) {
  if (notify_param == nullptr) {
    LOGS_DEFAULT(ERROR) << "ReleaseDmaDataCallback: notify_param is null";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  auto callback_info = reinterpret_cast<QnnBackendManager::FileMappingCallbackInfo_t*>(notify_param);

  if (callback_info->backend_manager == nullptr) {
    LOGS_DEFAULT(ERROR) << "ReleaseDmaDataCallback: QnnBackendManager is null";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  return callback_info->backend_manager->ReleaseDmaData(data_mem, callback_info->mapped_file_ptr);
}

// Use LOGS_DEFAULT here as this function will be called during destruction of QnnBackendManager
// At time of destruction, usage of logger_ will not be available and will result in a seg fault
Qnn_ErrorHandle_t QnnBackendManager::ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                                    void* mapped_base_ptr) {
  if (mapped_base_ptr == nullptr) {
    LOGS_DEFAULT(ERROR) << "Attempting to release DMA data for null memory mapped pointer";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  LOGS_DEFAULT(INFO) << "Releasing DMA data mapping for memory mapped pointer("
                     << mapped_base_ptr << "), address(" << data_mem.dmaBuffer.data
                     << "), size: (" << data_mem.memSize << ")";

  if (data_mem.dmaBuffer.data == nullptr || data_mem.memSize == 0) {
    LOGS_DEFAULT(ERROR) << "Mapping release request address must not be null and size must be > 0";
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // Deregister file mapped data from NPU regardless of file_mapped_weights_enabled_
  // as there may be file mapped data registered to the NPU prior to any mapping error
  void* unaligned_data_ptr = data_mem.dmaBuffer.data;
  rpcmem_library_->Api().register_buf(unaligned_data_ptr, data_mem.memSize, -1,
                                      rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

  auto fd = rpcmem_library_->Api().to_fd(unaligned_data_ptr);
  if (fd != -1) {
    LOGS_DEFAULT(ERROR) << "Failed to deregister buffer from RPCMEM: " << unaligned_data_ptr;
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }
  return QNN_SUCCESS;
}
#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

// callback required to add context handles to class list
// when using contextCreateFromBinaryListAsync()
static void ContextCreateAsyncCallback(Qnn_ContextHandle_t context,
                                       Qnn_GraphHandle_t /* graph */,
                                       const char* /* graph_name */,
                                       QnnContext_createFromBinaryAsyncNotifyType_t /* notify_type */,
                                       void* notify_param,
                                       Qnn_ErrorHandle_t /* status */) {
  auto qnn_backend_manager = SharedContext::GetInstance().GetSharedQnnBackendManager();

  if (context) {
    qnn_backend_manager->ProcessContextFromBinListAsync(context, notify_param);
  }
}

void QnnBackendManager::ProcessContextFromBinListAsync(Qnn_ContextHandle_t context, void* notifyParam) {
  std::lock_guard<std::mutex> guard(ep_context_handle_map_mutex_);
  if (!notifyParam) {
    LOGS(*logger_, WARNING) << "No known node names associated with context handle: " << context;
    return;
  }

  std::vector<std::string>* ep_node_names = reinterpret_cast<std::vector<std::string>*>(notifyParam);
  for (const auto& node_name : *ep_node_names) {
    if (!(ep_context_handle_map_.emplace(node_name, context).second)) {
      LOGS(*logger_, VERBOSE) << "Unable to map " << context << " to " << node_name;
    }
  }

  auto s = AddQnnContextHandle(context);
  if (s != Status::OK()) {
    LOGS(*logger_, WARNING) << "Unable to add context " << context;
  }
}

Status QnnBackendManager::GetFileSizeIfValid(const std::string& filepath,
                                             size_t& file_size) {
  std::error_code ec;
  ORT_RETURN_IF(!std::filesystem::exists(filepath, ec), "Context binary does not exist: ", filepath);
  ORT_RETURN_IF(ec, "Failed to read file: ", filepath,
                ", error: ", ec.message());

  auto size = std::filesystem::file_size(filepath, ec);
  ORT_RETURN_IF(ec, "Failed to retrieve size of file: ", filepath,
                ", error: ", ec.message());

  ORT_RETURN_IF(size == 0, "File is empty: ", filepath);
  ORT_RETURN_IF(size > SIZE_MAX, "File (", filepath, ") file size (", size,
                " bytes) exceeds maximum value of size_t for this platform (", SIZE_MAX, " bytes).");

  file_size = static_cast<size_t>(size);
  return Status::OK();
}

Status QnnBackendManager::ReadContextBinIfValid(const std::string& context_bin_filepath,
                                                std::vector<char>& buffer) {
  size_t buffer_size;
  ORT_RETURN_IF_ERROR(GetFileSizeIfValid(context_bin_filepath, buffer_size));

  buffer.resize(buffer_size);

  std::ifstream cache_file(context_bin_filepath.c_str(), std::ifstream::binary);
  ORT_RETURN_IF(!cache_file || !cache_file.good(), "Failed to read context binary from: ", context_bin_filepath);

  const auto& read_result = cache_file.read(buffer.data(), buffer_size);
  ORT_RETURN_IF(!read_result, "Failed to read contents from cached context file.");

  return Status::OK();
}

Status QnnBackendManager::CreateContextVtcmBackupBufferSharingEnabled(std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>& context_bin_map) {
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  QnnContext_Config_t context_config_resource_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t resource_sharing_custom_config;
  resource_sharing_custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES;
  resource_sharing_custom_config.shareResources = true;
  context_config_resource_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_resource_sharing.customConfig = &resource_sharing_custom_config;

  QnnHtpContext_CustomConfig_t context_config_resource_sharing_opt_type;
  context_config_resource_sharing_opt_type.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE;
  context_config_resource_sharing_opt_type.shareResOptType = SEQUENTIAL_WITHOUT_VA_OPTIMIZATION;
  QnnContext_Config_t resource_sharing_opt_type_config;
  resource_sharing_opt_type_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  resource_sharing_opt_type_config.customConfig = &context_config_resource_sharing_opt_type;

  QnnContext_Config_t context_config_weight_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t custom_config;
  custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  custom_config.weightSharingEnabled = true;
  context_config_weight_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_weight_sharing.customConfig = &custom_config;
#else
  LOGS(*logger_, WARNING) << "Called CreateContextVtcmBackupBufferSharingEnabled() but QNN API version is older than 2.26!";
#endif
  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  ORT_RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, context_priority_config));

  const QnnContext_Config_t* configs[] = {&context_priority_config,
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
                                          &context_config_resource_sharing,
                                          &resource_sharing_opt_type_config,
                                          &context_config_weight_sharing,
#endif
                                          nullptr};

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  if (file_mapped_weights_enabled_ && file_mapper_) {
    // Retry logic -- if context creation failed with file mapped weights, then retry with feature disabled
    auto res = CreateContextFromListAsyncWithCallback(configs, context_bin_map);
    if (!res.IsOK()) {
      LOGS(*logger_, WARNING) << res.ErrorMessage() << ". Retrying with feature disabled.";
    } else {
      return Status::OK();
    }
  }
#endif
  return CreateContextFromListAsync(configs, context_bin_map);
}

Status QnnBackendManager::CreateContextFromListAsync(const QnnContext_Config_t** configs,
                                                     std::unordered_map<std::string,
                                                                        std::unique_ptr<std::vector<std::string>>>& context_bin_map) {
  std::vector<QnnContext_Params_t> context_params_list;
  std::vector<QnnContext_ParamsV1_t> context_paramsv1_list;
  std::vector<const QnnContext_Params_t*> context_params_ptr_list;
  std::vector<std::vector<char>> buffer_list;

  context_params_list.reserve(context_bin_map.size());
  context_params_ptr_list.reserve(context_bin_map.size() + 1);

  for (auto& it : context_bin_map) {
    auto context_bin_filepath = it.first;

    std::vector<char> buffer;
    ORT_RETURN_IF_ERROR(ReadContextBinIfValid(context_bin_filepath, buffer));

    size_t buffer_size = buffer.size();
    buffer_list.push_back(std::move(buffer));

    QnnContext_ParamsV1_t context_params_v1 = {nullptr,
                                               buffer_list.back().data(),
                                               buffer_size,
                                               nullptr,
                                               ContextCreateAsyncCallback,
                                               it.second.get()};

    QnnContext_Params_t context_params = {QnnContext_ParamsVersion_t::QNN_CONTEXT_PARAMS_VERSION_1,
                                          {context_params_v1}};

    context_params_list.push_back(std::move(context_params));
    context_paramsv1_list.push_back(std::move(context_params_v1));
    context_params_ptr_list.push_back(&context_params_list.back());
  }
  context_params_ptr_list.push_back(nullptr);
  auto result = qnn_interface_.contextCreateFromBinaryListAsync(backend_handle_,
                                                                device_handle_,
                                                                context_params_ptr_list.data(),
                                                                configs,
                                                                nullptr);

  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to create context. Error: ", QnnErrorHandleToString(result), ", Code:", result);
  return Status::OK();
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
Status QnnBackendManager::CreateContextFromListAsyncWithCallback(const QnnContext_Config_t** configs,
                                                                 std::unordered_map<std::string,
                                                                                    std::unique_ptr<std::vector<std::string>>>& context_bin_map) {
  std::vector<QnnContext_Params_t> context_params_list;
  std::vector<QnnContext_ParamsV2_t> context_paramsv2_list;
  std::vector<Qnn_ContextBinaryCallback_t> context_callbacks_list;
  std::vector<const QnnContext_Params_t*> context_params_ptr_list;

  context_params_list.reserve(context_bin_map.size());
  context_paramsv2_list.reserve(context_bin_map.size());
  context_callbacks_list.reserve(context_bin_map.size());
  context_params_ptr_list.reserve(context_bin_map.size() + 1);

  for (auto& it : context_bin_map) {
    auto context_bin_filepath = it.first;

    size_t buffer_size;
    ORT_RETURN_IF_ERROR(GetFileSizeIfValid(context_bin_filepath, buffer_size));

    void* buffer;
    ORT_RETURN_IF_ERROR(file_mapper_->GetContextBinMappedMemoryPtr(context_bin_filepath, &buffer));

    auto notify_param_ptr = std::make_unique<FileMappingCallbackInfo_t>(buffer, buffer_size, this);

    Qnn_ContextBinaryCallback_t context_file_map_callbacks;
    context_file_map_callbacks.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
    context_file_map_callbacks.dmaBufferCallback.version = QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
    context_file_map_callbacks.dmaBufferCallback.v1.dataProvide = MapDmaDataCallback;
    context_file_map_callbacks.dmaBufferCallback.v1.dataRelease = ReleaseDmaDataCallback;
    context_file_map_callbacks.dmaBufferCallback.v1.notifyParam = reinterpret_cast<void*>(notify_param_ptr.get());

    file_mapping_notify_params_.push_back(std::move(notify_param_ptr));
    context_callbacks_list.push_back(std::move(context_file_map_callbacks));

    // Callbacks require QnnContext_ParamsV2_t which is new to QNN API 2.32
    QnnContext_ParamsV2_t context_params_v2 = {nullptr,
                                               buffer,
                                               buffer_size,
                                               nullptr,
                                               ContextCreateAsyncCallback,
                                               it.second.get(),
                                               &context_callbacks_list.back()};

    QnnContext_Params_t context_params = {QnnContext_ParamsVersion_t::QNN_CONTEXT_PARAMS_VERSION_2,
                                          {}};

    context_paramsv2_list.push_back(std::move(context_params_v2));

    context_params.v2 = &context_paramsv2_list.back();
    context_params_list.push_back(std::move(context_params));
    context_params_ptr_list.push_back(&(context_params_list.back()));
  }
  context_params_ptr_list.push_back(nullptr);
  auto result = qnn_interface_.contextCreateFromBinaryListAsync(backend_handle_,
                                                                device_handle_,
                                                                context_params_ptr_list.data(),
                                                                configs,
                                                                nullptr);

  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to create context with file mapping enabled. Error: ",
                QnnErrorHandleToString(result), ", Code:", result);
  return Status::OK();
}
#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

Status QnnBackendManager::SetContextPriority(ContextPriority context_priority) {
  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  ORT_RETURN_IF_ERROR(SetQnnContextConfig(context_priority, context_priority_config));

  QnnContext_Config_t* configs[] = {&context_priority_config, nullptr};
  for (const auto& context_handle : contexts_) {
    auto result = qnn_interface_.contextSetConfig(context_handle, (const QnnContext_Config_t**)configs);
    ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to set context priority for context handle: ", context_handle);
  }

  return Status::OK();
}

Status QnnBackendManager::ResetContextPriority() {
  return SetContextPriority(context_priority_);
}

Status QnnBackendManager::CreateContext(bool enable_htp_weight_sharing, bool enable_htp_extended_udma_mode) {
  if (true == context_created_) {
    LOGS_DEFAULT(INFO) << "Context created already.";
    return Status::OK();
  }

  QnnContext_Config_t context_config_weight_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t custom_config;
  custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  custom_config.weightSharingEnabled = enable_htp_weight_sharing;
  context_config_weight_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_weight_sharing.customConfig = &custom_config;

  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  ORT_RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, context_priority_config));

  QnnContext_Config_t context_config_extended_udma = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t udma_custom_config;
  udma_custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA;
  udma_custom_config.useExtendedUdma = enable_htp_extended_udma_mode;
  context_config_extended_udma.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_extended_udma.customConfig = &udma_custom_config;

  const QnnContext_Config_t* npu_context_configs[] = {&context_priority_config,
                                                      &context_config_weight_sharing,
                                                      &context_config_extended_udma,
                                                      nullptr};

  const QnnContext_Config_t* empty_context_configs[] = {nullptr};

  const QnnContext_Config_t** configs = nullptr;
  switch (GetQnnBackendType()) {
    case QnnBackendType::HTP:
    case QnnBackendType::DSP:
      configs = npu_context_configs;
      break;
    case QnnBackendType::GPU:
    case QnnBackendType::SERIALIZER:
      configs = nullptr;
      break;
    default:
      configs = empty_context_configs;
      break;
  }

  // Not all serialization backends allow for hardware configs to be applied.
  if (qnn_serializer_config_ && !qnn_serializer_config_->SupportsArbitraryGraphConfigs()) {
    configs = nullptr;
  }

  Qnn_ContextHandle_t context = nullptr;
  Qnn_ErrorHandle_t result = 0;

  result = qnn_interface_.contextCreate(backend_handle_,
                                        device_handle_,
                                        configs,
                                        &context);

  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to create context. Error: ", QnnErrorHandleToString(result), ", Code:", result);

  ORT_RETURN_IF_ERROR(AddQnnContextHandle(context));

  context_created_ = true;
  return Status::OK();
}

Status QnnBackendManager::ReleaseContext() {
  if (false == context_created_) {
    return Status::OK();
  }

  // release QNN context handles
  contexts_.clear();
  context_map_.clear();

  context_created_ = false;
  return Status::OK();
}

std::unique_ptr<unsigned char[]> QnnBackendManager::GetContextBinaryBuffer(uint64_t& written_buffer_size) {
  if (nullptr == qnn_interface_.contextGetBinarySize ||
      nullptr == qnn_interface_.contextGetBinary) {
    LOGS(*logger_, ERROR) << "Failed to get valid function pointer.";
    return nullptr;
  }
  ORT_ENFORCE(contexts_.size() > 0, "No valid QNN context!");
  uint64_t required_buffer_size(0);
  // Generate all graphs in one single context
  Qnn_ErrorHandle_t rt = qnn_interface_.contextGetBinarySize(contexts_[0], &required_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    LOGS(*logger_, ERROR) << "Failed to get QNN context binary size. Error: " << QnnErrorHandleToString(rt);
    return nullptr;
  }

  std::unique_ptr<unsigned char[]> context_buffer = std::make_unique<unsigned char[]>(required_buffer_size);
  if (nullptr == context_buffer) {
    LOGS(*logger_, ERROR) << "Failed to allocate buffer for context cache.";
    return nullptr;
  }

  rt = qnn_interface_.contextGetBinary(contexts_[0],
                                       reinterpret_cast<void*>(context_buffer.get()),
                                       required_buffer_size,
                                       &written_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    LOGS(*logger_, ERROR) << "Failed to get context binary. Error: " << QnnErrorHandleToString(rt);
    return nullptr;
  }

  if (required_buffer_size < written_buffer_size) {
    LOGS(*logger_, ERROR) << "Context written buffer size: " << written_buffer_size
                          << " exceeds allocated buffer size: " << required_buffer_size;
    return nullptr;
  }

  LOGS(*logger_, VERBOSE) << "Get context binary buffer succeed.";
  return context_buffer;
}

Status QnnBackendManager::GetMaxSpillFillBufferSize(unsigned char* buffer,
                                                    uint64_t buffer_length,
                                                    uint64_t& max_spill_fill_buffer_size) {
  max_spill_fill_buffer_size = 0;
  // spill fill starts from 2.28
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)
  bool result = nullptr == qnn_sys_interface_.systemContextCreate ||
                nullptr == qnn_sys_interface_.systemContextGetBinaryInfo ||
                nullptr == qnn_sys_interface_.systemContextFree;
  ORT_RETURN_IF(result, "Failed to get valid function pointer.");

  QnnSystemContext_Handle_t sys_ctx_handle = nullptr;
  auto rt = qnn_sys_interface_.systemContextCreate(&sys_ctx_handle);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to create system handle.");

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size{0};
  rt = qnn_sys_interface_.systemContextGetBinaryInfo(sys_ctx_handle,
                                                     static_cast<void*>(buffer),
                                                     buffer_length,
                                                     &binary_info,
                                                     &binary_info_size);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to get context binary info.");

  // binary_info life cycle is here
  // Binary info to graph info
  // retrieve Qnn graph info from binary info
  ORT_RETURN_IF(nullptr == binary_info, "Qnn cached binary info is nullptr.");
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graph_count = binary_info->contextBinaryInfoV3.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV3.graphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graph_count = binary_info->contextBinaryInfoV2.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV2.graphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graph_count = binary_info->contextBinaryInfoV1.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV1.graphs;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported context binary info version.");
  }

  for (uint32_t i = 0; i < graph_count; ++i) {
    if (graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      auto htp_graph_info = reinterpret_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(graphs_info[i].graphInfoV3.graphBlobInfo);
      if (htp_graph_info->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
        auto spill_fill_buffer_size = htp_graph_info->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
        max_spill_fill_buffer_size = spill_fill_buffer_size > max_spill_fill_buffer_size ? spill_fill_buffer_size : max_spill_fill_buffer_size;
      } else {
        LOGS(*logger_, VERBOSE) << "Unknown context binary graph info blob version.";
      }
    } else if (graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 ||
               graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      LOGS(*logger_, VERBOSE) << "Skip retrieve spill file buffer size, it is not supported with graph info v1 & v2.";
    } else {
      LOGS(*logger_, VERBOSE) << "Unknown context binary graph info version.";
    }
  }
#else
  ORT_UNUSED_PARAMETER(buffer);
  ORT_UNUSED_PARAMETER(buffer_length);
#endif

  LOGS(*logger_, VERBOSE) << "Get max spill fill buffer size completed.";
  return Status::OK();
}

Status QnnBackendManager::LoadCachedQnnContextFromBuffer(char* buffer, uint64_t buffer_length,
                                                         const std::string& context_bin_filepath,
                                                         std::string node_name,
                                                         QnnModelLookupTable& qnn_models,
                                                         int64_t max_spill_fill_size) {
  bool result = nullptr == qnn_sys_interface_.systemContextCreate ||
                nullptr == qnn_sys_interface_.systemContextGetBinaryInfo ||
                nullptr == qnn_sys_interface_.systemContextFree;
  ORT_RETURN_IF(result, "Failed to get valid function pointer.");

  void* bin_buffer = nullptr;
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  if (file_mapped_weights_enabled_) {
    ORT_RETURN_IF(!file_mapper_, "Attemping to use File Mapping feature but file_mapper_ is uninitialized");

    ORT_RETURN_IF_ERROR(GetFileSizeIfValid(context_bin_filepath, buffer_length));

    ORT_RETURN_IF(buffer_length == 0, "Context bin has a size of 0 bytes: ", context_bin_filepath);
    ORT_RETURN_IF_ERROR(file_mapper_->GetContextBinMappedMemoryPtr(context_bin_filepath, &bin_buffer));

  } else {
    ORT_RETURN_IF(buffer == nullptr, "Attempting to load QNN context from buffer but buffer is null");
    bin_buffer = static_cast<void*>(buffer);
  }
#else
  bin_buffer = static_cast<void*>(buffer);
#endif

  QnnSystemContext_Handle_t sys_ctx_handle = nullptr;
  auto rt = qnn_sys_interface_.systemContextCreate(&sys_ctx_handle);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to create system handle.");

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size{0};
  rt = qnn_sys_interface_.systemContextGetBinaryInfo(sys_ctx_handle,
                                                     bin_buffer,
                                                     buffer_length,
                                                     &binary_info,
                                                     &binary_info_size);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to get context binary info.");

  // binary_info life cycle is here
  // Binary info to graph info
  // retrieve Qnn graph info from binary info
  ORT_RETURN_IF(nullptr == binary_info, "Qnn cached binary info is nullptr.");
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graph_count = binary_info->contextBinaryInfoV1.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV1.graphs;
  }
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 15)  // starts from 2.22
  else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graph_count = binary_info->contextBinaryInfoV2.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV2.graphs;
  }
#endif
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)  // starts from 2.28
  else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graph_count = binary_info->contextBinaryInfoV3.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV3.graphs;
  }
#endif
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported context binary info version.");
  }

  ORT_RETURN_IF(graph_count < 1 || graphs_info == nullptr, "Failed to get graph info from Qnn cached context.");
  LOGS(*logger_, VERBOSE) << "Graph count from QNN context: " << graph_count;

  Qnn_ContextHandle_t context = nullptr;
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  if (vtcm_backup_buffer_sharing_enabled_) {
    if (ep_context_handle_map_.find(node_name) != ep_context_handle_map_.end()) {
      context = ep_context_handle_map_.at(node_name);
    }
    ORT_RETURN_IF(nullptr == context, "Failed to retrieve context for ", node_name);

  } else {
#endif
    QnnContext_Config_t qnn_context_config = QNN_CONTEXT_CONFIG_INIT;
    ORT_RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, qnn_context_config));

    // Register spill fill buffer for multi context
    QnnContext_Config_t spill_fill_config = QNN_CONTEXT_CONFIG_INIT;

    // The spill fill buffer is available since 2.28, API version starts from 2.21
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)
    QnnHtpContext_CustomConfig_t custom_config;
    custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    QnnHtpContext_GroupRegistration_t group_info;
    size_t current_contexts_size = GetQnnContextSize();
    // set to 0x0 (new group) if this is the first context, otherwise point to the first context handle
    // note that we already move the context with max spill fill size to the beginning of the list
    group_info.firstGroupHandle = (max_spill_fill_size > 0 && current_contexts_size > 0) ? GetQnnContext(0) : 0x0;
    group_info.maxSpillFillBuffer = max_spill_fill_size;  // Max spill-fill buffer across contexts. Must be >0
    custom_config.groupRegistration = group_info;
    spill_fill_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    spill_fill_config.customConfig = &custom_config;

#endif

    QnnContext_Config_t* spill_fill_config_pointer = max_spill_fill_size > 0 ? &spill_fill_config : nullptr;
    LOGS(*logger_, VERBOSE) << "Max spill fill buffer size:" << max_spill_fill_size;

    const QnnContext_Config_t* context_configs[] = {&qnn_context_config, spill_fill_config_pointer, nullptr};

    ORT_RETURN_IF(nullptr == qnn_interface_.contextCreateFromBinary,
                  "Invalid function pointer for contextCreateFromBinary.");

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
    Qnn_ContextBinaryCallback_t callbacks;
    if (file_mapped_weights_enabled_ && file_mapper_) {
      ORT_RETURN_IF(nullptr == qnn_interface_.contextCreateFromBinaryWithCallback,
                    "Invalid function pointer for contextCreateFromBinaryWithCallback.");

      auto notify_param_ptr = std::make_unique<FileMappingCallbackInfo_t>(bin_buffer, buffer_length, this);

      callbacks.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
      callbacks.dmaBufferCallback.version = QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
      callbacks.dmaBufferCallback.v1.dataProvide = MapDmaDataCallback;
      callbacks.dmaBufferCallback.v1.dataRelease = ReleaseDmaDataCallback;
      callbacks.dmaBufferCallback.v1.notifyParam = reinterpret_cast<void*>(notify_param_ptr.get());

      file_mapping_notify_params_.push_back(std::move(notify_param_ptr));
    }
#else
  ORT_UNUSED_PARAMETER(context_bin_filepath);
#endif

    qnn::profile::ProfilingInfo profiling_info;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (ProfilingEnabled()) {
      profiling_info.start_time = qnn::utils::GetTimeStampInUs();
    }
#endif

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
    std::vector<char> backup_buffer;
    if (file_mapped_weights_enabled_ && file_mapper_) {
      rt = qnn_interface_.contextCreateFromBinaryWithCallback(backend_handle_,
                                                              device_handle_,
                                                              context_configs,
                                                              &callbacks,
                                                              bin_buffer,
                                                              buffer_length,
                                                              &context,
                                                              profile_backend_handle_,
                                                              NULL);

      if (rt != QNN_SUCCESS) {
        LOGS(*logger_, WARNING) << "Failed to create context with file mapping enabled. Error: "
                                << QnnErrorHandleToString(rt) << ", Code : " << rt
                                << ". Retrying with feature disabled.";

        // Read context bin from file since file mapping has failed
        ORT_RETURN_IF_ERROR(ReadContextBinIfValid(context_bin_filepath, backup_buffer));

        bin_buffer = static_cast<void*>(backup_buffer.data());
      }
    }
#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

    if (!file_mapped_weights_enabled_ || rt != QNN_SUCCESS) {
      rt = qnn_interface_.contextCreateFromBinary(backend_handle_,
                                                  device_handle_,
                                                  context_configs,
                                                  bin_buffer,
                                                  buffer_length,
                                                  &context,
                                                  profile_backend_handle_);
    }

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (ProfilingEnabled()) {
      profiling_info.stop_time = qnn::utils::GetTimeStampInUs();
      profiling_info.method_type = ProfilingMethodType::CREATE_FROM_BINARY;
      profiling_info.graph_name = node_name;
    }
#endif

    ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to create context from binary. Error code: ", rt);
    ORT_RETURN_IF_ERROR(AddQnnContextHandle(context));

    ORT_RETURN_IF_ERROR(ExtractBackendProfilingInfo(profiling_info));

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  }
#endif

  if (1 == graph_count) {
    // in case the EPContext node is generated from script
    // the graph name from the context binary may not match the EPContext node name
    auto qnn_model = std::make_unique<qnn::QnnModel>(this);
    ORT_RETURN_IF_ERROR(qnn_model->DeserializeGraphInfoFromBinaryInfo(graphs_info[0], context));
    qnn_models.emplace(node_name, std::move(qnn_model));
  } else {
    for (uint32_t i = 0; i < graph_count; ++i) {
      auto qnn_model = std::make_unique<qnn::QnnModel>(this);
      ORT_RETURN_IF_ERROR(qnn_model->DeserializeGraphInfoFromBinaryInfo(graphs_info[i], context));
      qnn_models.emplace(graphs_info[i].graphInfoV1.graphName, std::move(qnn_model));
    }
  }

  qnn_sys_interface_.systemContextFree(sys_ctx_handle);
  sys_ctx_handle = nullptr;
  context_created_ = true;

  LOGS(*logger_, VERBOSE) << "Load from cached QNN Context completed.";
  return Status::OK();
}

// need to load system lib if load from Qnn context binary
// or generate Qnn context binary is enabled -- to get the max spill fill buffer size
Status QnnBackendManager::SetupBackend(const logging::Logger& logger,
                                       bool load_from_cached_context,
                                       bool need_load_system_lib,
                                       bool share_ep_contexts,
                                       bool enable_vtcm_backup_buffer_sharing,
                                       bool enable_file_mapped_weights,
                                       std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library,
                                       std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>& context_bin_map,
                                       bool enable_htp_extended_udma_mode) {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (backend_setup_completed_) {
    LOGS(logger, VERBOSE) << "Backend setup already!";

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
    if (vtcm_backup_buffer_sharing_enabled_) {
      // If a context bin filepath has not been processed yet,
      // then a new context must be created for the set of context bins
      auto first_mapping_it = ep_context_handle_map_.find(context_bin_map.begin()->first);
      if (first_mapping_it == ep_context_handle_map_.end()) {
        LOGS(logger, VERBOSE) << "Creating context for new set of context binaries";
        return CreateContextVtcmBackupBufferSharingEnabled(context_bin_map);
      }

      LOGS(logger, VERBOSE) << "Mapping contexts to new EP main context nodes";

      for (auto& it : context_bin_map) {
        auto context_bin_filepath = it.first;
        auto ep_node_names = *(it.second);

        auto context = ep_context_handle_map_.at(context_bin_filepath);
        for (auto node_name : ep_node_names) {
          ep_context_handle_map_.emplace(node_name, context);
        }
      }
    }
#endif
    return Status::OK();
  }

  vtcm_backup_buffer_sharing_enabled_ = enable_vtcm_backup_buffer_sharing;

  Status status = Status::OK();
  if (!qnn_serializer_config_) {
    status = LoadBackend();
  } else {
    status = LoadQnnSerializerBackend();
  }

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  // Backend is determined after LoadBackend() or LoadQnnSerializerBackend()
  if (enable_file_mapped_weights && !file_mapper_ && GetQnnBackendType() == QnnBackendType::HTP) {
    ORT_RETURN_IF(!rpcmem_library, "RPCMem Library is required for file mapping but is uninitialized.");
    rpcmem_library_ = rpcmem_library;
    file_mapped_weights_enabled_ = true;
    file_mapper_ = std::make_unique<WindowsFileMapper>(logger);
  }
#else
  ORT_UNUSED_PARAMETER(enable_file_mapped_weights);
  ORT_UNUSED_PARAMETER(rpcmem_library);
#endif

  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "LoadBackend succeed.";
  }

  if (status.IsOK() && (load_from_cached_context || need_load_system_lib)) {
    status = LoadQnnSystemLib();
  }

  if (status.IsOK()) {
    sdk_build_version_ = GetBackendBuildId();
    LOGS(logger, VERBOSE) << "Backend build version: "
                          << sdk_build_version_;
  }

  if (status.IsOK()) {
    status = InitializeQnnLog(logger);
  }
  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "SetLogger succeed.";
  }

  if (status.IsOK()) {
    status = InitializeBackend();
  }
  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "InitializeBackend succeed.";
  }

  if (status.IsOK()) {
    status = CreateDevice();
  }
  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "CreateDevice succeed.";
  }

  if (status.IsOK()) {
    status = InitializeProfiling();
  }
  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "InitializeProfiling succeed.";
  }

  if (status.IsOK()) {
    ORT_RETURN_IF_ERROR(LoadOpPackage());
    LOGS(logger, VERBOSE) << "LoadOpPackage succeed.";
  }

  bool enable_htp_weight_sharing = false;
  if (share_ep_contexts && !load_from_cached_context) {
#if defined(__aarch64__) || defined(_M_ARM64)
    LOGS(logger, WARNING) << "Weight sharing only available with offline generation on x64 platform, not work on real device.";
#else
    enable_htp_weight_sharing = true;
#endif
  }

  if (status.IsOK() && (vtcm_backup_buffer_sharing_enabled_ || !load_from_cached_context)) {
    status = vtcm_backup_buffer_sharing_enabled_ ? CreateContextVtcmBackupBufferSharingEnabled(context_bin_map)
                                                 : CreateContext(enable_htp_weight_sharing, enable_htp_extended_udma_mode);

    if (status.IsOK()) {
      LOGS(logger, VERBOSE) << "CreateContext succeed.";
    }
  }

  if (status.IsOK()) {
    LOGS(logger, VERBOSE) << "QNN SetupBackend succeed";
    backend_setup_completed_ = true;
  } else {
    LOGS_DEFAULT(WARNING) << "Failed to setup so cleaning up";
    ReleaseResources();
  }

  return status;
}

Status QnnBackendManager::CreateHtpPowerCfgId(uint32_t device_id, uint32_t core_id, uint32_t& htp_power_config_id) {
  // This function is called in QNN EP's OnRunStart() even if QNN backend setup failed and the model is assigned
  // to a different EP. Therefore, we have to check that backend setup actually completed before trying to
  // create an HTP power config ID. Otherwise, this causes a segfault because the QNN backend lib is unloaded.
  ORT_RETURN_IF_NOT(backend_setup_completed_, "Cannot create HTP power config ID if backend setup is not complete.");
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  ORT_RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
                "HTP infra type = ", htp_infra->infraType, ", which is not perf infra type.");
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;
  // Get power client id
  status = htp_perf_infra.createPowerConfigId(device_id, core_id, &htp_power_config_id);
  ORT_RETURN_IF(QNN_SUCCESS != status, "createPowerConfigId failed.");

  return Status::OK();
}

Status QnnBackendManager::SetHtpPowerConfigs(uint32_t htp_power_config_client_id,
                                             HtpPerformanceMode htp_performance_mode,
                                             uint32_t rpc_polling_time,
                                             uint32_t rpc_control_latency) {
  // This function is called in QNN EP's OnRunStart() even if QNN backend setup failed and the model is assigned
  // to a different EP. Therefore, we have to check that backend setup actually completed before trying to
  // set an HTP power config ID. Otherwise, this causes a segfault because the QNN backend lib is unloaded.
  ORT_RETURN_IF_NOT(backend_setup_completed_, "Cannot set HTP power config ID if backend setup is not complete.");

  ORT_RETURN_IF_ERROR(htp_power_config_manager_.AddRpcPollingTime(rpc_polling_time));
  ORT_RETURN_IF_ERROR(htp_power_config_manager_.AddRpcControlLatency(rpc_control_latency));
  ORT_RETURN_IF_ERROR(htp_power_config_manager_.AddHtpPerformanceMode(htp_performance_mode, htp_power_config_client_id));
  ORT_RETURN_IF_ERROR(htp_power_config_manager_.SetPowerConfig(htp_power_config_client_id, GetQnnInterface()));

  return Status::OK();
}

Status QnnBackendManager::SetPerThreadHtpPowerConfigs(const std::thread::id& thread_id, bool pre_run) {
  PerThreadHtpPowerConfigs_t htp_power_configs;
  if (!GetPerThreadHtpPowerConfigMapping(thread_id, htp_power_configs)) {
    return Status::OK();
  }

  auto htp_power_config_id = htp_power_configs.power_config_id;
  if (pre_run) {
    if (htp_power_configs.pre_run_perf_mode.has_value()) {
      ORT_RETURN_IF_ERROR(htp_power_config_manager_.AddHtpPerformanceMode(*htp_power_configs.pre_run_perf_mode,
                                                                          htp_power_config_id));
    }

    if (htp_power_configs.rpc_control_latency.has_value()) {
      ORT_RETURN_IF_ERROR(htp_power_config_manager_.AddRpcControlLatency(*htp_power_configs.rpc_control_latency));
    }

    if (htp_power_configs.rpc_polling_time.has_value()) {
      ORT_RETURN_IF_ERROR(htp_power_config_manager_.AddRpcPollingTime(*htp_power_configs.rpc_polling_time));
    }
  } else if (htp_power_configs.post_run_perf_mode.has_value()) {
    ORT_RETURN_IF_ERROR(htp_power_config_manager_.AddHtpPerformanceMode(*htp_power_configs.post_run_perf_mode,
                                                                        htp_power_config_id));
  }

  ORT_RETURN_IF_ERROR(htp_power_config_manager_.SetPowerConfig(htp_power_config_id, GetQnnInterface()));

  return Status::OK();
}

Status QnnBackendManager::AddPerThreadHtpPowerConfigMapping(const std::thread::id& thread_id,
                                                            const PerThreadHtpPowerConfigs_t& htp_power_configs) {
  std::lock_guard<std::mutex> lock(per_thread_power_configs_mutex_);

  auto res = per_thread_power_configs_.find(thread_id);
  ORT_RETURN_IF(res != per_thread_power_configs_.end(), "Trying to set HtpPowerConfigs for thread id ", thread_id,
                " but one already exists!");

  per_thread_power_configs_.emplace(thread_id, std::move(htp_power_configs));

  return Status::OK();
}

bool QnnBackendManager::GetPerThreadHtpPowerConfigMapping(const std::thread::id& thread_id,
                                                          PerThreadHtpPowerConfigs_t& htp_power_configs) {
  std::lock_guard<std::mutex> lock(per_thread_power_configs_mutex_);

  auto it = per_thread_power_configs_.find(thread_id);
  if (it == per_thread_power_configs_.end()) {
    return false;
  }

  htp_power_configs = it->second;
  return true;
}

void QnnBackendManager::RemovePerThreadHtpPowerConfigMapping(const std::thread::id& thread_id) {
  std::lock_guard<std::mutex> lock(per_thread_power_configs_mutex_);
  per_thread_power_configs_.erase(thread_id);
}

Status QnnBackendManager::DestroyHTPPowerConfigID(uint32_t htp_power_config_id) {
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  ORT_RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
                "HTP infra type = ", htp_infra->infraType, ", which is not perf infra type.");
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

  Qnn_ErrorHandle_t destroy_ret = htp_perf_infra.destroyPowerConfigId(htp_power_config_id);
  ORT_RETURN_IF(QNN_SUCCESS != destroy_ret, "destroyPowerConfigId failed.");
  return Status::OK();
}

Status QnnBackendManager::TerminateQnnLog() {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (logger_ == nullptr) {
    return Status::OK();
  }

  if (nullptr != qnn_interface_.logFree && nullptr != log_handle_) {
    auto ret_val = qnn_interface_.logFree(log_handle_);

    // Reset QNN log handle to nullptr so other threads that are waiting on logger_recursive_mutex_ know it was freed.
    log_handle_ = nullptr;
    ORT_RETURN_IF(QNN_SUCCESS != ret_val,
                  "Unable to terminate logging in the backend.");
  }

  return Status::OK();
}

void QnnBackendManager::ReleaseResources() {
  auto result = ReleaseContext();
  if (Status::OK() != result) {
    LOGS_DEFAULT(ERROR) << "Failed to ReleaseContext: " << result.ErrorMessage();
  }

  result = ReleaseProfilehandle();
  if (Status::OK() != result) {
    LOGS_DEFAULT(ERROR) << "Failed to ReleaseProfilehandle: " << result.ErrorMessage();
  }

  result = ReleaseDevice();
  if (Status::OK() != result) {
    LOGS_DEFAULT(ERROR) << "Failed to ReleaseDevice: " << result.ErrorMessage();
  }

  result = ShutdownBackend();
  if (Status::OK() != result) {
    LOGS_DEFAULT(ERROR) << "Failed to ShutdownBackend: " << result.ErrorMessage();
  }

  result = TerminateQnnLog();
  if (Status::OK() != result) {
    LOGS_DEFAULT(ERROR) << "Failed to TerminateQnnLog: " << result.ErrorMessage();
  }

  if (backend_lib_handle_) {
    result = UnloadLib(backend_lib_handle_);
    if (Status::OK() != result) {
      LOGS_DEFAULT(ERROR) << "Failed to unload backend library: " << result.ErrorMessage();
    }
  }

  backend_setup_completed_ = false;
  return;
}

Status QnnBackendManager::ExtractBackendProfilingInfo(qnn::profile::ProfilingInfo& profiling_info) {
  if (ProfilingLevel::OFF == profiling_level_merge_ || ProfilingLevel::INVALID == profiling_level_merge_) {
    return Status::OK();
  }

  bool tracelogging_provider_ep_enabled = false;
#ifdef _WIN32
  auto& provider = QnnTelemetry::Instance();
  if (provider.IsEnabled()) {
    auto level = provider.Level();
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)) != 0 && level >= 5) {
      tracelogging_provider_ep_enabled = true;
    }
  }
#endif  // defined(_WIN32)

  // ETW disabled previously, but enabled now
  if (ProfilingLevel::INVALID == profiling_level_etw_ && tracelogging_provider_ep_enabled) {
    LOGS(*logger_, ERROR) << "ETW disabled previously, but enabled now. Can't do the switch! Won't output any profiling.";
    return Status::OK();
  }

  // ETW enabled previously, but disabled now
  if (ProfilingLevel::INVALID != profiling_level_etw_ && !tracelogging_provider_ep_enabled) {
    LOGS(*logger_, ERROR) << "ETW enabled previously, but disabled now. Can't do the switch! Won't output any profiling.";
    return Status::OK();
  }

  ORT_RETURN_IF(!tracelogging_provider_ep_enabled && profiling_file_path_.empty(),
                "Need to specify a CSV file via provider option profiling_file_path if ETW not enabled.");

  ORT_RETURN_IF(nullptr == profile_backend_handle_, "Backend profile handle not valid.");

  LOGS(*logger_, VERBOSE) << "Extracting profiling events for graph " << profiling_info.graph_name;
  const QnnProfile_EventId_t* profile_events{nullptr};
  uint32_t num_events{0};
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetEvents(profile_backend_handle_, &profile_events, &num_events);
  if (qnn_serializer_config_) {  // Using QNN Saver or IR backend
    // QNN SDK 2.28.2 returns QNN_SAVER_ERROR_DUMMY_RETVALUE, but previous QNN versions return QNN_PROFILE_NO_ERROR.
    // We accept both values.
    ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result && QNN_SAVER_ERROR_DUMMY_RETVALUE != result,
                  "Failed to get profile events. Error: ", QnnErrorHandleToString(result));
  } else {
    ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile events. Error: ", QnnErrorHandleToString(result));
  }

  if (num_events > 0) {
    LOGS(*logger_, VERBOSE) << "profile_events: " << profile_events << " num_events: " << num_events;

    bool backendSupportsExtendedEventData = false;
    Qnn_ErrorHandle_t resultPropertyHasCapability =
        qnn_interface_.propertyHasCapability(QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT);
    uint16_t errorCodePropertyHasCapability = static_cast<uint16_t>(resultPropertyHasCapability & 0xFFFF);
    if (errorCodePropertyHasCapability == QNN_PROPERTY_SUPPORTED) {
      LOGS(*logger_, VERBOSE) << "The QNN backend supports extended event data.";
      backendSupportsExtendedEventData = true;
    } else {
      LOGS(*logger_, VERBOSE) << "The QNN backend does not support extended event data.";
    }

    profiling_info.csv_output_filepath = profiling_file_path_;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    profiling_info.num_events = num_events;
#endif

    profile::Serializer profile_writer(profiling_info,
                                       qnn_sys_interface_,
                                       tracelogging_provider_ep_enabled);
    if (!profiling_file_path_.empty()) {
      ORT_RETURN_IF_ERROR(profile_writer.InitCsvFile());
    }

    for (size_t event_idx = 0; event_idx < num_events; event_idx++) {
      ORT_RETURN_IF_ERROR(
          ExtractProfilingEvent(*(profile_events + event_idx), "ROOT", profile_writer,
                                backendSupportsExtendedEventData));
      ORT_RETURN_IF_ERROR(
          ExtractProfilingSubEvents(*(profile_events + event_idx), profile_writer,
                                    backendSupportsExtendedEventData));
    }
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    ORT_RETURN_IF_ERROR(profile_writer.SerializeEventsToQnnLog());
#endif

    if (!profiling_file_path_.empty()) {
      LOGS(*logger_, VERBOSE) << "Wrote QNN profiling events (" << num_events << ") to file ("
                              << profiling_file_path_ << ")";
    }

    if (tracelogging_provider_ep_enabled) {
      LOGS(*logger_, VERBOSE) << "Wrote QNN profiling events (" << num_events << ") to ETW";
    }
  }

  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingSubEvents(
    QnnProfile_EventId_t profile_event_id,
    profile::Serializer& profile_writer,
    bool useExtendedEventData) {
  const QnnProfile_EventId_t* profile_sub_events{nullptr};
  uint32_t num_sub_events{0};
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile sub events. Error: ", QnnErrorHandleToString(result));

  if (num_sub_events > 0) {
    LOGS(*logger_, VERBOSE) << "profile_sub_events: " << profile_sub_events << " num_sub_events: " << num_sub_events;

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    QnnSystemProfile_ProfileEventV1_t* parent_system_event = nullptr;
    parent_system_event = profile_writer.GetParentSystemEvent(profile_event_id);
    if (parent_system_event == nullptr) {
      parent_system_event = profile_writer.GetSystemEventPointer(profile_event_id);
      profile_writer.AddSubEventList(num_sub_events, parent_system_event);
    }
#endif

    for (size_t sub_event_idx = 0; sub_event_idx < num_sub_events; sub_event_idx++) {
      QnnProfile_EventId_t subevent_id = *(profile_sub_events + sub_event_idx);

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED

      ORT_RETURN_IF_ERROR(profile_writer.SetParentSystemEvent(subevent_id, parent_system_event));

#endif

      ORT_RETURN_IF_ERROR(
          ExtractProfilingEvent(subevent_id, "SUB-EVENT", profile_writer, useExtendedEventData));
      ORT_RETURN_IF_ERROR(
          ExtractProfilingSubEvents(subevent_id, profile_writer, useExtendedEventData));
    }

    LOGS(*logger_, VERBOSE) << "Wrote QNN profiling sub events (" << num_sub_events << ")";
  }

  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingEvent(
    QnnProfile_EventId_t profile_event_id,
    const std::string& event_level,
    profile::Serializer& profile_writer,
    bool useExtendedEventData) {
  if (useExtendedEventData) {
    return ExtractProfilingEventExtended(profile_event_id, event_level, profile_writer);
  } else {
    return ExtractProfilingEventBasic(profile_event_id, event_level, profile_writer);
  }
}

Status QnnBackendManager::ExtractProfilingEventBasic(
    QnnProfile_EventId_t profile_event_id,
    const std::string& event_level,
    profile::Serializer& profile_writer) {
  QnnProfile_EventData_t event_data;
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetEventData(profile_event_id, &event_data);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(result & 0xFFFF);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode)));

  ORT_RETURN_IF_ERROR(profile_writer.ProcessEvent(profile_event_id, event_level, event_data));

  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingEventExtended(
    QnnProfile_EventId_t profile_event_id,
    const std::string& event_level,
    profile::Serializer& profile_writer) {
  QnnProfile_ExtendedEventData_t event_data_extended;
  auto resultGetExtendedEventData = qnn_interface_.profileGetExtendedEventData(profile_event_id, &event_data_extended);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(resultGetExtendedEventData & 0xFFFF);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != errorCode, "Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode)));

  ORT_RETURN_IF_ERROR(profile_writer.ProcessExtendedEvent(profile_event_id, event_level, event_data_extended));

  return Status::OK();
}

const char* QnnBackendManager::QnnProfileErrorToString(QnnProfile_Error_t error) {
  switch (error) {
    case QNN_PROFILE_NO_ERROR:
      return "QNN_PROFILE_NO_ERROR";
    case QNN_PROFILE_ERROR_UNSUPPORTED:
      return "QNN_PROFILE_ERROR_UNSUPPORTED";
    case QNN_PROFILE_ERROR_INVALID_ARGUMENT:
      return "QNN_PROFILE_ERROR_INVALID_ARGUMENT";
    case QNN_PROFILE_ERROR_MEM_ALLOC:
      return "QNN_PROFILE_ERROR_MEM_ALLOC";
    case QNN_PROFILE_ERROR_INVALID_HANDLE:
      return "QNN_PROFILE_ERROR_INVALID_HANDLE";
    case QNN_PROFILE_ERROR_HANDLE_IN_USE:
      return "QNN_PROFILE_ERROR_HANDLE_IN_USE";
    case QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT:
      return "QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT";
    default:
      return "UNKNOWN_ERROR";
  }
}

std::string QnnBackendManager::QnnErrorHandleToString(Qnn_ErrorHandle_t error) {
  return utils::GetQnnErrorMessage(qnn_interface_, error);
}

QnnBackendManager::~QnnBackendManager() {
  ReleaseResources();
}

void* QnnBackendManager::LoadLib(const char* file_name, int flags, std::string& error_msg) {
#ifdef _WIN32
  DWORD as_is, to_be;
  bool loaded_before = false;

  if (!file_name || ::strlen(file_name) == 0) {
    error_msg = "filename is null or empty";
    return nullptr;
  }

  // POSIX asks one of symbol resolving approaches:
  // NOW or LAZY must be specified
  if (!(flags & static_cast<int>(DlOpenFlag::DL_NOW))) {
    error_msg = "flags must include DL_NOW";
    return nullptr;
  }

  HANDLE cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, nullptr, 0, &as_is) == 0) {
    error_msg = "enumerate modules failed before loading module";
    return nullptr;
  }

  HMODULE mod;
  auto file_path = std::filesystem::path(file_name);
  if (!file_path.is_absolute()) {
    // construct an absolute path from ORT runtime path + file_name and check whether it exists.
    const Env& env = GetDefaultEnv();
    auto pathstring = env.GetRuntimePath() + ToPathString(file_name);
    auto absolute_path = pathstring.c_str();
    if (std::filesystem::exists(std::filesystem::path(absolute_path))) {
      // load library from absolute path and search for dependencies there.
      mod = LoadLibraryExW(absolute_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    } else {
      // use default dll search order for file_name.
      mod = LoadLibraryExA(file_name, nullptr, 0);
    }
  } else {
    // file_name represents an absolute path.
    // load library from absolute path and search for dependencies there.
    mod = LoadLibraryExA(file_name, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  }
  if (!mod) {
    error_msg = "load library failed";
    return nullptr;
  }

  if (EnumProcessModules(cur_proc, nullptr, 0, &to_be) == 0) {
    error_msg = "enumerate modules failed after loading module";
    FreeLibrary(mod);
    return nullptr;
  }

  if (as_is == to_be) {
    loaded_before = true;
  }

  // (not loaded_before) and DL_LOCAL means this lib was not loaded yet
  // add it into the local set
  //
  // If loaded_before and DL_LOCAL, means this lib was already loaded
  // 2 cases here for how it was loaded before:
  // a. with DL_LOCAL, just ignore since it was already in local set
  // b. with DL_GLOBAL, POSIX asks it in global, ignore it, too
  if ((!loaded_before) && (flags & static_cast<int>(DlOpenFlag::DL_LOCAL))) {
    mod_handles_.insert(mod);
  }

  // once callers ask for global, needs to be in global thereafter
  // so the lib should be removed from local set
  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    mod_handles_.erase(mod);
  }

  return static_cast<void*>(mod);
#else
  ORT_UNUSED_PARAMETER(error_msg);
  int real_flags = 0;

  if (flags & static_cast<int>(DlOpenFlag::DL_NOW)) {
    real_flags |= RTLD_NOW;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_LOCAL)) {
    real_flags |= RTLD_LOCAL;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    real_flags |= RTLD_GLOBAL;
  }

  return ::dlopen(file_name, real_flags);
#endif
}

Status QnnBackendManager::UnloadLib(void* handle) {
  if (!handle) {
    return Status::OK();
  }

#ifdef _WIN32
  HMODULE mod = static_cast<HMODULE>(handle);

  if (FreeLibrary(mod) == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to free library.");
  }
  mod_handles_.erase(mod);
#else
  auto rt = ::dlclose(handle);
  if (rt != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to free library.");
  }
#endif  // defined(_WIN32)

  return Status::OK();
}

void* QnnBackendManager::LibFunction(void* handle, const char* symbol, std::string& error_msg) {
#ifdef _WIN32
  FARPROC sym_addr = nullptr;
  DWORD size, size_needed;
  HMODULE mod = 0;

  if ((!handle) || (!symbol)) {
    return nullptr;
  }

  HANDLE cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, nullptr, 0, &size) == 0) {
    error_msg = "enumerate modules failed before memory allocation";
    return nullptr;
  }

  HMODULE* mod_list = static_cast<HMODULE*>(malloc(size));
  if (!mod_list) {
    error_msg = "malloc failed";
    return nullptr;
  }

  if (EnumProcessModules(cur_proc, mod_list, size, &size_needed) == 0) {
    error_msg = "enumerate modules failed after memory allocation";
    free(mod_list);
    return nullptr;
  }

  // DL_DEFAULT needs to bypass those modules with DL_LOCAL flag
  if (handle == DL_DEFAULT) {
    for (size_t i = 0; i < (size / sizeof(HMODULE)); i++) {
      auto iter = mod_handles_.find(mod_list[i]);
      if (iter != mod_handles_.end()) {
        continue;
      }
      // once find the first non-local module with symbol
      // return its address here to avoid unnecessary looping
      sym_addr = GetProcAddress(mod_list[i], symbol);
      if (sym_addr) {
        free(mod_list);
        return *(void**)(&sym_addr);
      }
    }
  } else {
    mod = static_cast<HMODULE>(handle);
  }

  free(mod_list);
  sym_addr = GetProcAddress(mod, symbol);
  if (!sym_addr) {
    error_msg = "can't resolve symbol";
    return NULL;
  }

  return *(void**)(&sym_addr);
#else
  ORT_UNUSED_PARAMETER(error_msg);
  if (handle == DL_DEFAULT) {
    return ::dlsym(RTLD_DEFAULT, symbol);
  }

  return ::dlsym(handle, symbol);
#endif
}

Status QnnBackendManager::AddQnnContextHandle(Qnn_ContextHandle_t raw_context_handle) {
  ORT_RETURN_IF(logger_ == nullptr, "logger_ should be set.");

  auto free_context_handle = [this, &logger = *logger_](Qnn_ContextHandle_t raw_context_handle) {
    const auto free_result = qnn_interface_.contextFree(raw_context_handle, nullptr);
    if (free_result != QNN_CONTEXT_NO_ERROR) {
      LOGS(logger, ERROR) << "qnn_interface.contextFree() failed: "
                          << utils::GetVerboseQnnErrorMessage(qnn_interface_, free_result);
    }
  };

  // take ownership of `raw_context_handle`
  auto context_handle = UniqueQnnContextHandle(raw_context_handle, free_context_handle);
  auto mem_handle_manager = std::make_unique<QnnContextMemHandleManager>(GetQnnInterface(), raw_context_handle,
                                                                         *logger_);

  auto context_handle_record = std::make_shared<QnnContextHandleRecord>();
  context_handle_record->context_handle = std::move(context_handle);
  context_handle_record->mem_handles = std::move(mem_handle_manager);

  const bool inserted = context_map_.try_emplace(raw_context_handle, std::move(context_handle_record)).second;
  ORT_RETURN_IF_NOT(inserted, "QNN context was already added: ", raw_context_handle);

  contexts_.push_back(raw_context_handle);

  return Status::OK();
}

Status QnnBackendManager::GetOrRegisterContextMemHandle(Qnn_ContextHandle_t context_handle,
                                                        void* shared_memory_address,
                                                        const Qnn_Tensor_t& qnn_tensor,
                                                        Qnn_MemHandle_t& mem_handle) {
  // Multi-threading situations to consider:
  // 1) Shared memory allocation is being freed in another thread while we are processing `shared_memory_address`.
  //    This implies incorrect usage as the memory is being freed while it is still in use. Let's assume this won't
  //    happen.
  // 2) The shared memory allocation clean up function is being run from another thread while the
  //    QnnContextHandleRecord or QnnBackendManager objects are being destroyed.
  //    Usage of weak_ptrs from the clean up function should ensure that those objects are only accessed while they are
  //    in scope.

  const auto context_handle_record_it = context_map_.find(context_handle);
  ORT_RETURN_IF_NOT(context_handle_record_it != context_map_.end(), "QNN context not found: ", context_handle);

  auto& context_handle_record = context_handle_record_it->second;
  auto& context_mem_handle_manager = context_handle_record->mem_handles;

  bool did_register{};
  ORT_RETURN_IF_ERROR(context_mem_handle_manager->GetOrRegister(shared_memory_address, qnn_tensor,
                                                                mem_handle, did_register));

  if (did_register) {
    HtpSharedMemoryAllocator::AllocationCleanUpFn unregister_mem_handle =
        [&logger = *logger_,
         shared_memory_address,
         weak_backend_manager = weak_from_this(),
         weak_context_handle_record = std::weak_ptr{context_handle_record}](
            void* /* allocation_base_address */) {
          // Lock QnnBackendManager shared_ptr to ensure that QNN interface is still valid.
          auto backend_manager = weak_backend_manager.lock();
          if (!backend_manager) {
            return;
          }

          // Lock QnnContextHandleRecord shared_ptr to ensure that QNN context handle is still valid.
          auto context_handle_record = weak_context_handle_record.lock();
          if (!context_handle_record) {
            return;
          }

          auto& context_mem_handle_manager = context_handle_record->mem_handles;

          auto unregister_status = context_mem_handle_manager->Unregister(shared_memory_address);
          if (!unregister_status.IsOK()) {
            LOGS(logger, ERROR) << "Failed to unregister shared memory mem handle for address: "
                                << shared_memory_address << ", error: " << unregister_status.ErrorMessage();
          }
        };

    ORT_RETURN_IF_ERROR(HtpSharedMemoryAllocator::AddAllocationCleanUp(shared_memory_address,
                                                                       std::move(unregister_mem_handle)));
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
