//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_backend_manager.h"

#include <filesystem>
#include <fstream>
#include <gsl/gsl>
#include <string>

#include "CPU/QnnCpuCommon.h"
#include "DSP/QnnDspCommon.h"
#include "GPU/QnnGpuCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpSystemContext.h"
#include "IR/QnnIrCommon.h"
#include "IR/QnnIrGraph.h"
#include "QnnOpDef.h"
#include "Saver/QnnSaver.h"
#include "Saver/QnnSaverCommon.h"

#include "core/providers/qnn-abi/builder/qnn_configs_helper.h"
#include "core/providers/qnn-abi/builder/qnn_model.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/qnn_allocator.h"
#include "core/providers/qnn-abi/qnn_telemetry.h"
#include "core/providers/qnn-abi/shared_context.h"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

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

Ort::Status ReadBinaryFromFile(const std::string& file_path, uint8_t* buffer, size_t buffer_size) {
  RETURN_IF(nullptr == buffer, "Binary buffer is nullptr");
  std::ifstream in(file_path, std::ifstream::binary);
  RETURN_IF(!in, ("Failed to open input file: " + file_path).c_str());
  RETURN_IF(!in.read(reinterpret_cast<char*>(buffer), buffer_size),
            ("Failed to read the contents of: " + file_path).c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::ParseLoraConfig(std::string lora_config_path) {
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("Acquiring the QnnInterface " + lora_config_path).c_str());

  // QNN Lora Config file format should be a single line, with the graph name first,
  // followed by the qnn lora context binary path, separated by a semicolon (;)
  // Example: <graph_name>;<binary_path>
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("Loading Lora Config " + lora_config_path).c_str());
  std::ifstream file(lora_config_path);
  std::string line;

  if (file.is_open()) {
    if (std::getline(file, line)) {
      std::istringstream ss(line);
      std::string graph_name;
      std::string lora_adapter_bin_path;

      if (std::getline(ss, graph_name, ';') && std::getline(ss, lora_adapter_bin_path)) {
        size_t buffer_size = std::filesystem::file_size(lora_adapter_bin_path.c_str());

        RETURN_IF(0 == buffer_size, "Received path to an empty file. Nothing to deserialize.");
        std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(buffer_size);
        void* voidBufferPtr = static_cast<void*>(buffer.get());
        QnnContext_Buffer_t contextBuffer{QNN_CONTEXT_BUFFER_VERSION_1,
                                          {QNN_CONTEXTMEMTYPE_RAW, {{voidBufferPtr, buffer_size}}}};

        auto status = ReadBinaryFromFile(lora_adapter_bin_path,
                                         reinterpret_cast<uint8_t*>(buffer.get()),
                                         buffer_size);

        RETURN_IF(!status.IsOK(), "Failed to read binary data.");
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
          RETURN_IF(QNN_SUCCESS != context_apply_binary_section_rt, "Failed to apply binary section.");
          break;
        }
        RETURN_IF_NOT(graph_retrieve_success,
                      ("Failed to retrieve graph: " + graph_name + " and apply binary section.").c_str());
      }
    }
    file.close();
  } else {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Unable to load Lora Config " + lora_config_path).c_str());
  }

  return Ort::Status();
}

template <typename F, class T>
Ort::Status QnnBackendManager::GetQnnInterfaceProvider(const char* lib_path,
                                                       const char* interface_provider_name,
                                                       void** backend_lib_handle,
                                                       Qnn_Version_t req_version,
                                                       T** interface_provider) {
  std::string error_msg;
  *backend_lib_handle = LoadLib(lib_path,
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_GLOBAL),
                                error_msg);
  RETURN_IF(nullptr == *backend_lib_handle, ("Unable to load backend, error: " + error_msg + " " + DlError()).c_str());

  // Get QNN Interface providers
  F GetInterfaceProviders{nullptr};
  GetInterfaceProviders = ResolveSymbol<F>(*backend_lib_handle, interface_provider_name, logger_);
  RETURN_IF(nullptr == GetInterfaceProviders, "Failed to get QNN providers!");

  T** interface_providers{nullptr};
  uint32_t num_providers{0};

  auto result = GetInterfaceProviders((const T***)&interface_providers, &num_providers);
  RETURN_IF((QNN_SUCCESS != result || nullptr == *interface_providers || 0 == num_providers),
            "Failed to get QNN providers.");

  if (skip_qnn_version_check_) {
    // When skipping version check, use the first available provider.
    *interface_provider = interface_providers[0];
    return Ort::Status();
  }

  bool found_valid_interface{false};
  for (size_t pIdx = 0; pIdx < num_providers; pIdx++) {
    Qnn_Version_t interface_version = GetQnnInterfaceApiVersion(interface_providers[pIdx]);

    std::ostringstream oss;
    oss << lib_path << " interface version: " << interface_version.major << "."
        << interface_version.minor << "." << interface_version.patch;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

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

  RETURN_IF_NOT(found_valid_interface, ("Unable to find a valid interface for " + std::string(lib_path)).c_str());

  return Ort::Status();
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

Ort::Status QnnBackendManager::LoadBackend() {
#if defined(__aarch64__) && defined(__linux__)
  // QNN requires ADSP_LIBRARY_PATH to be set in order to find skel libs on Linux
  static std::once_flag set_adsp_path_once;

  std::call_once(set_adsp_path_once, []() {
    constexpr std::string_view kAdspLibraryPathEnvVar{"ADSP_LIBRARY_PATH"};
    const char* existingPath = getenv(kAdspLibraryPathEnvVar.data());
    if (existingPath != nullptr) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                  ORT_LOGGING_LEVEL_WARNING,
                  ("Using existing ADSP_LIBRARY_PATH setting of " +
                   std::string(existingPath) + ", which may cause the HTP backend to fail.")
                      .c_str());
      return;
    }

    std::filesystem::path qnnLibPath(OrtGetRuntimePath());
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_WARNING,
                ("Setting " + std::string(kAdspLibraryPathEnvVar) + " = " + qnnLibPath.string()).c_str());
    setenv(kAdspLibraryPathEnvVar.data(), qnnLibPath.c_str(), 1);
  });
#endif

  QnnInterface_t* backend_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t, QnnInterface_t>(backend_path_.c_str(),
                                                                                         "QnnInterface_getProviders",
                                                                                         &backend_lib_handle_,
                                                                                         {QNN_API_VERSION_MAJOR,
                                                                                          QNN_API_VERSION_MINOR,
                                                                                          QNN_API_VERSION_PATCH},
                                                                                         &backend_interface_provider)));
  qnn_interface_ = backend_interface_provider->QNN_INTERFACE_VER_NAME;
  auto backend_id = backend_interface_provider->backendId;
  SetQnnBackendType(backend_id);

  Qnn_Version_t backend_interface_version = GetQnnInterfaceApiVersion(backend_interface_provider);
  std::ostringstream oss;
  oss << "Found valid interface, version: " << backend_interface_version.major
      << "." << backend_interface_version.minor << "." << backend_interface_version.patch
      << " backend provider name: " << backend_interface_provider->providerName
      << " backend id: " << backend_id;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, oss.str().c_str());

  return Ort::Status();
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
Ort::Status QnnBackendManager::LoadQnnSerializerBackend() {
  void* backend_lib_handle = nullptr;

  // Helper that unloads the intended backend library handle when the `unload_backend_lib` variable
  // goes out of scope. Similar to `defer` in other languages.
  auto unload_backend_lib = gsl::finally([&] {
    if (backend_lib_handle != nullptr) {
      auto result = UnloadLib(backend_lib_handle);
      if (!result.IsOK()) {
        ORT_CXX_API_THROW("Failed to unload backend library.", ORT_EP_FAIL);
      }
    }
  });

  // Load the intended backend (e.g., HTP, CPU) to ensure it is valid and to get its type.
  QnnInterface_t* backend_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t, QnnInterface_t>(backend_path_.c_str(),
                                                                                         "QnnInterface_getProviders",
                                                                                         &backend_lib_handle,
                                                                                         {QNN_API_VERSION_MAJOR,
                                                                                          QNN_API_VERSION_MINOR,
                                                                                          QNN_API_VERSION_PATCH},
                                                                                         &backend_interface_provider)));

  // Set the "intended" backend type so that QNN builders still make the expected QNN API calls.
  auto backend_id = backend_interface_provider->backendId;
  SetQnnBackendType(backend_id);

  // Load the serializer backend and set it as the activate backend.
  QnnInterface_t* serializer_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t, QnnInterface_t>(
      qnn_serializer_config_->GetBackendPath().c_str(),
      "QnnInterface_getProviders",
      &backend_lib_handle_,  // NOTE: QnnSaver/Ir library handle is set
      {QNN_API_VERSION_MAJOR,
       QNN_API_VERSION_MINOR,
       QNN_API_VERSION_PATCH},
      &serializer_interface_provider)));
  qnn_interface_ = serializer_interface_provider->QNN_INTERFACE_VER_NAME;  // NOTE: QnnSaver/Ir will provide the interfaces

  Qnn_Version_t backend_interface_version = GetQnnInterfaceApiVersion(backend_interface_provider);
  Qnn_Version_t serializer_interface_version = GetQnnInterfaceApiVersion(serializer_interface_provider);

  std::ostringstream oss1;
  oss1 << "Using QnnSaver/Ir version: " << serializer_interface_version.major << "."
       << serializer_interface_version.minor << "." << serializer_interface_version.patch
       << " provider name : " << serializer_interface_provider->providerName;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, oss1.str().c_str());

  std::ostringstream oss2;
  oss2 << "Intended backend provider name: " << backend_interface_provider->providerName
       << " backend id: " << backend_id
       << " interface version: " << backend_interface_version.major
       << "." << backend_interface_version.minor << "." << backend_interface_version.patch;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, oss2.str().c_str());

  return Ort::Status();
}

Ort::Status QnnBackendManager::LoadQnnSystemLib() {
  if (system_lib_loaded_) {
    return Ort::Status();
  }

#ifdef _WIN32
  std::string system_lib_file = "QnnSystem.dll";
#else
  std::string system_lib_file = "libQnnSystem.so";
#endif  // #ifdef _WIN32
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Loading QnnSystem lib");
  std::filesystem::path lib_file_path(backend_path_.c_str());
  std::string sys_file_path(lib_file_path.remove_filename().string() + system_lib_file);
  QnnSystemInterface_t* system_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnSystemInterfaceGetProvidersFn_t, QnnSystemInterface_t>(
      sys_file_path.c_str(),
      "QnnSystemInterface_getProviders",
      &system_lib_handle_,
      {QNN_SYSTEM_API_VERSION_MAJOR,
       QNN_SYSTEM_API_VERSION_MINOR,
       QNN_SYSTEM_API_VERSION_PATCH},
      &system_interface_provider)));
  Qnn_Version_t system_interface_version = GetQnnInterfaceApiVersion(system_interface_provider);
  qnn_sys_interface_ = system_interface_provider->QNN_SYSTEM_INTERFACE_VER_NAME;

  std::ostringstream oss;
  oss << "Found valid system interface, version: " << system_interface_version.major
      << "." << system_interface_version.minor
      << " backend provider name: " << system_interface_provider->providerName;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, oss.str().c_str());

  system_lib_loaded_ = true;
  return Ort::Status();
}

void QnnLogging(const char* format,
                QnnLog_Level_t level,
                uint64_t timestamp,
                va_list argument_parameter) {
  ORT_UNUSED_PARAMETER(level);
  ORT_UNUSED_PARAMETER(timestamp);

  if (!OrtLoggingManager::HasDefaultLogger()) {
    return;
  }

  // QNN-EP COPY START
  // There is an unknown bug in Ort::Logger::LogFormattedMessage which causes crashes.
  // Below implementations are directly copied from core/common/logging/capture.cc to create formatted string
  // and leverage Ort::Logger::LogMessage instead.

  static constexpr auto kTruncatedWarningText = "[...truncated...]";
  static constexpr int kMaxMessageSize = 2048;
  char message_buffer[kMaxMessageSize];
  const auto message = gsl::make_span(message_buffer);

  bool error = false;
  bool truncated = false;

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) && !defined(__GNUC__))
  errno = 0;
  const int nbrcharacters = vsnprintf_s(message.data(), message.size(), _TRUNCATE, format, argument_parameter);
  if (nbrcharacters < 0) {
    error = errno != 0;
    truncated = !error;
  }
#else
  const int nbrcharacters = vsnprintf(message.data(), message.size(), format, argument_parameter);
  error = nbrcharacters < 0;
  truncated = (nbrcharacters >= 0 && static_cast<size_t>(nbrcharacters) > message.size());
#endif

  std::ostringstream stream;
  if (error) {
    stream << "\n\tERROR LOG MSG NOTIFICATION: Failure to successfully parse the message";
    stream << '"' << format << '"' << std::endl;
  } else if (truncated) {
    stream << message.data() << kTruncatedWarningText;
  } else {
    stream << message.data();
  }
  // QNN-EP COPY END

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, stream.str().c_str());
}

Ort::Status QnnBackendManager::InitializeQnnLog() {
  // Set Qnn log level align with Ort log level
  auto ort_log_level = logger_.GetLoggingSeverityLevel();
  QnnLog_Level_t qnn_log_level = MapOrtSeverityToQNNLogLevel(ort_log_level);
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Set Qnn log level: " + std::to_string(qnn_log_level)).c_str());

  // NOTE: Even if logCreate() fails and QNN does not return a valid log_handle_, QNN may still
  // call the QnnLogging() callback. So, we have to make sure that QnnLogging() can handle calls
  // in which ORT logging is not available.
  Qnn_ErrorHandle_t result = qnn_interface_.logCreate(QnnLogging, qnn_log_level, &log_handle_);

  if (result != QNN_SUCCESS) {
    switch (result) {
      case QNN_COMMON_ERROR_NOT_SUPPORTED:
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Logging not supported in the QNN backend.");
        break;
      case QNN_LOG_ERROR_INVALID_ARGUMENT:
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Invalid argument provided to QnnLog_create.");
        break;
      case QNN_LOG_ERROR_MEM_ALLOC:
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Memory allocation error during QNN logging initialization.");
        break;
      case QNN_LOG_ERROR_INITIALIZATION:
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Initialization of logging failed in the QNN backend.");
        break;
      default:
        ORT_CXX_LOG(logger_,
                    ORT_LOGGING_LEVEL_WARNING,
                    "Unknown error occurred while initializing logging in the QNN backend.");
        break;
    }
  }

  RETURN_IF(QNN_BACKEND_NO_ERROR != result,
            ("Failed to initialize logging in the QNN backend. Error: " + QnnErrorHandleToString(result)).c_str());
  return Ort::Status();
}

QnnLog_Level_t QnnBackendManager::MapOrtSeverityToQNNLogLevel(OrtLoggingLevel ort_log_level) {
  // Map ORT log severity to Qnn log level
  switch (ort_log_level) {
    case ORT_LOGGING_LEVEL_VERBOSE: {
      switch ((GetQnnBackendType())) {
        case QnnBackendType::GPU:
          // Currently GPU needs this log level to work.
          // This switch will be removed once this is resolved.
          return QNN_LOG_LEVEL_DEBUG;
        default:
          return QNN_LOG_LEVEL_VERBOSE;
      }
    }
    case ORT_LOGGING_LEVEL_INFO:
      return QNN_LOG_LEVEL_INFO;
    case ORT_LOGGING_LEVEL_WARNING:
      return QNN_LOG_LEVEL_WARN;
    case ORT_LOGGING_LEVEL_ERROR:
    case ORT_LOGGING_LEVEL_FATAL:
    default:
      return QNN_LOG_LEVEL_ERROR;
  }
}

Ort::Status QnnBackendManager::ResetQnnLogLevel(std::optional<OrtLoggingLevel> ort_log_level) {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (!backend_setup_completed_) {
    return Ort::Status();
  }
  RETURN_IF(nullptr == log_handle_, "Unable to update QNN Log Level. Invalid QNN log handle.");

  OrtLoggingLevel actual_log_level = ort_log_level.has_value() ? *ort_log_level : logger_.GetLoggingSeverityLevel();
  QnnLog_Level_t qnn_log_level = MapOrtSeverityToQNNLogLevel(actual_log_level);

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("Updating Qnn log level to: " + std::to_string(qnn_log_level)).c_str());

  // Use the QnnLog_setLogLevel API to set the new log level
  Qnn_ErrorHandle_t result = qnn_interface_.logSetLogLevel(log_handle_, qnn_log_level);
  if (QNN_SUCCESS != result) {
    if (result == QNN_LOG_ERROR_INVALID_ARGUMENT) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Invalid log level argument provided to QnnLog_setLogLevel.");
    } else if (result == QNN_LOG_ERROR_INVALID_HANDLE) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Invalid log handle provided to QnnLog_setLogLevel.");
    }
  }
  RETURN_IF(QNN_BACKEND_NO_ERROR != result,
            ("Failed to set log level in Qnn backend. Error: " + QnnErrorHandleToString(result)).c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::InitializeBackend() {
  if (true == backend_initialized_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Backend initialized already.");
    return Ort::Status();
  }

  Qnn_ErrorHandle_t result = qnn_interface_.backendCreate(log_handle_, (const QnnBackend_Config_t**)backend_config_, &backend_handle_);
  RETURN_IF(QNN_BACKEND_NO_ERROR != result,
            ("Failed to initialize backend. Error: " + QnnErrorHandleToString(result)).c_str());

  backend_initialized_ = true;
  return Ort::Status();
}

Ort::Status QnnBackendManager::ShutdownBackend() {
  if (false == backend_initialized_) {
    return Ort::Status();
  }

  if (nullptr != qnn_interface_.backendFree) {
    RETURN_IF(QNN_BACKEND_NO_ERROR != qnn_interface_.backendFree(backend_handle_), "Failed to shutdown backend!");
  }

  backend_initialized_ = false;

  return Ort::Status();
}

bool QnnBackendManager::IsDevicePropertySupported() {
  if (nullptr != qnn_interface_.propertyHasCapability) {
    auto rt = qnn_interface_.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == rt || QNN_PROPERTY_ERROR_UNKNOWN_KEY == rt) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Device property not supported or unknown to backend.");
      return false;
    }
  }

  return true;
}

Ort::Status QnnBackendManager::CreateDevice() {
  if (true == device_created_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Device initialized already.");
    return Ort::Status();
  }

  // Create device if its property supported
  if (!IsDevicePropertySupported()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Skip to create device.");
    return Ort::Status();
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

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Create device.");
  if (nullptr != qnn_interface_.deviceCreate) {
    Qnn_ErrorHandle_t result = qnn_interface_.deviceCreate(log_handle_, device_configs_builder.GetQnnConfigs(), &device_handle_);
    if (QNN_SUCCESS != result) {
      return MAKE_EP_FAIL(("Failed to create device. Error: " + QnnErrorHandleToString(result)).c_str());
    }
  }
  device_created_ = true;

  return Ort::Status();
}

Ort::Status QnnBackendManager::ReleaseDevice() {
  if (false == device_created_) {
    return Ort::Status();
  }

  if (nullptr != qnn_interface_.deviceFree) {
    Qnn_ErrorHandle_t result = qnn_interface_.deviceFree(device_handle_);
    if (QNN_SUCCESS != result) {
      return MAKE_EP_FAIL(("Failed to release device. Error: " + QnnErrorHandleToString(result)).c_str());
    }
  }

  device_created_ = false;

  return Ort::Status();
}

Ort::Status QnnBackendManager::InitializeProfiling() {
  profiling_level_merge_ = profiling_level_;
  // use profiling level from ETW if ETW is enabled
  if (profiling_level_etw_ != ProfilingLevel::INVALID) {
    profiling_level_merge_ = profiling_level_etw_;
  }

  if (ProfilingLevel::OFF == profiling_level_merge_ || ProfilingLevel::INVALID == profiling_level_merge_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Profiling turned off.");
    return Ort::Status();
  }

  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  bool enable_optrace = false;
  if (ProfilingLevel::BASIC == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to basic.");
  } else if (ProfilingLevel::DETAILED == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to detailed.");
  } else if (ProfilingLevel::OPTRACE == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    enable_optrace = true;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to optrace.");
  }

  Qnn_ErrorHandle_t result = qnn_interface_.profileCreate(backend_handle_, qnn_profile_level, &profile_backend_handle_);
  RETURN_IF(QNN_PROFILE_NO_ERROR != result,
            ("Failed to create QNN profile! Error: " + QnnErrorHandleToString(result)).c_str());

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  profiling_enabled_ = true;
  RETURN_IF_ERROR(LoadQnnSystemLib());

  if (enable_optrace) {
    QnnProfile_Config_t optrace_config = QNN_PROFILE_CONFIG_INIT;
    optrace_config.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
    optrace_config.enableOptrace = enable_optrace;

    const QnnProfile_Config_t* profile_configs[] = {&optrace_config, nullptr};
    result = qnn_interface_.profileSetConfig(profile_backend_handle_, profile_configs);

    RETURN_IF(QNN_PROFILE_NO_ERROR != result,
              ("Failed to enable op trace! Error: " + QnnErrorHandleToString(result)).c_str());
  }
#else
  if (enable_optrace) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                "Profiling level set to optrace, but QNN SDK Version is older than 2.29.0. "
                "Profiling level will be set to detailed instead.");
  }
#endif

  return Ort::Status();
}

Ort::Status QnnBackendManager::ReleaseProfilehandle() {
  // Free Profiling object if it was created
  if (nullptr != profile_backend_handle_) {
    RETURN_IF(QNN_PROFILE_NO_ERROR != qnn_interface_.profileFree(profile_backend_handle_),
              "Could not free backend profile handle!");
  }
  profile_backend_handle_ = nullptr;

  return Ort::Status();
}

Ort::Status QnnBackendManager::SetProfilingLevelETW(ProfilingLevel profiling_level_etw_param) {
  if (profiling_level_etw_ != profiling_level_etw_param) {
    profiling_level_etw_ = profiling_level_etw_param;

    auto result = ReleaseProfilehandle();
    if (!result.IsOK()) {
      ORT_CXX_API_THROW("Failed to ReleaseProfilehandle for previous QNN profiling", ORT_EP_FAIL);
    }

    result = InitializeProfiling();
    if (!result.IsOK()) {
      ORT_CXX_API_THROW("Failed to Re-InitializeProfiling for QNN ETW profiling", ORT_EP_FAIL);
    }
  }
  return Ort::Status();
}

Ort::Status SetQnnContextConfig(ContextPriority context_priority, QnnContext_Config_t& qnn_context_config) {
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
      return MAKE_EP_FAIL("Invalid Qnn context priority.");
    }
    default:
      qnn_context_config.priority = QNN_PRIORITY_NORMAL;
  }  // switch

  return Ort::Status();
}

// callback required to add context handles to class list
// when using contextCreateFromBinaryListAsync()
void ContextCreateAsyncCallback(Qnn_ContextHandle_t context,
                                Qnn_GraphHandle_t graph,
                                const char* graphName,
                                QnnContext_createFromBinaryAsyncNotifyType_t notifyType,
                                void* notifyParam,
                                Qnn_ErrorHandle_t status) {
  auto qnn_backend_manager = SharedContext::GetInstance().GetSharedQnnBackendManager();

  if (context) {
    qnn_backend_manager->ProcessContextFromBinListAsync(context, notifyParam);
  }

  if (nullptr == graphName || graph || notifyType || status) {
    // Avoid compilation unused var warning error
  }
}

void QnnBackendManager::ProcessContextFromBinListAsync(Qnn_ContextHandle_t context, void* notifyParam) {
  std::ostringstream context_ss;
  context_ss << context;

  std::lock_guard<std::mutex> guard(ep_context_handle_map_mutex_);
  if (!notifyParam) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                ("No known node names associated with context handle: " + context_ss.str()).c_str());
    return;
  }

  std::vector<std::string>* ep_node_names = reinterpret_cast<std::vector<std::string>*>(notifyParam);
  for (const auto& node_name : *ep_node_names) {
    if (!(ep_context_handle_map_.emplace(node_name, context).second)) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Unable to map " + context_ss.str() + " to " + node_name).c_str());
    }
  }

  auto s = AddQnnContextHandle(context);
  if (!s.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING, ("Unable to add context " + context_ss.str()).c_str());
  }
}

Ort::Status QnnBackendManager::CreateContextVtcmBackupBufferSharingEnabled(
    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>& context_bin_map) {
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
  LOGS(logger_, WARNING) << "Called CreateContextVtcmBackupBufferSharingEnabled() but QNN API version is older than 2.26!";
#endif
  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, context_priority_config));

  const QnnContext_Config_t* configs[] = {&context_priority_config,
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
                                          &context_config_resource_sharing,
                                          &resource_sharing_opt_type_config,
                                          &context_config_weight_sharing,
#endif
                                          nullptr};

  std::vector<QnnContext_Params_t> context_params_list;
  std::vector<QnnContext_ParamsV1_t> context_paramsv1_list;
  std::vector<const QnnContext_Params_t*> context_params_ptr_list;
  std::vector<std::unique_ptr<char[]>> buffer_list;

  context_params_list.reserve(context_bin_map.size());
  context_params_ptr_list.reserve(context_bin_map.size() + 1);

  for (auto& it : context_bin_map) {
    auto context_bin_filepath = it.first;

    std::ifstream cache_file(context_bin_filepath.c_str(), std::ifstream::binary);
    RETURN_IF(!cache_file || !cache_file.good(),
              ("Failed to retrieve context binary from: " + context_bin_filepath).c_str());

    cache_file.seekg(0, cache_file.end);
    size_t buffer_size = static_cast<size_t>(cache_file.tellg());
    RETURN_IF(0 == buffer_size, "Empty cache file encountered.");

    cache_file.seekg(0, cache_file.beg);
    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(buffer_size);
    RETURN_IF(nullptr == buffer, "Failed to allocate memory for cache file.");
    const auto& read_result = cache_file.read(buffer.get(), buffer_size);
    RETURN_IF(!read_result, "Failed to read contents from cached context file.");

    cache_file.close();
    QnnContext_ParamsV1_t context_params_v1 = {nullptr,
                                               buffer.get(),
                                               buffer_size,
                                               nullptr,
                                               ContextCreateAsyncCallback,
                                               it.second.get()};

    QnnContext_Params_t context_params = {QnnContext_ParamsVersion_t::QNN_CONTEXT_PARAMS_VERSION_1,
                                          {context_params_v1}};

    buffer_list.push_back(std::move(buffer));
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

  context_params_ptr_list.clear();
  context_paramsv1_list.clear();
  context_params_list.clear();
  buffer_list.clear();

  RETURN_IF(QNN_CONTEXT_NO_ERROR != result,
            ("Failed to create context. Error: " + QnnErrorHandleToString(result)).c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::SetContextPriority(ContextPriority context_priority) {
  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  RETURN_IF_ERROR(SetQnnContextConfig(context_priority, context_priority_config));

  QnnContext_Config_t* configs[] = {&context_priority_config, nullptr};
  for (const auto& context_handle : contexts_) {
    auto result = qnn_interface_.contextSetConfig(context_handle, (const QnnContext_Config_t**)configs);
    RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to set context priority for context handle.");
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::ResetContextPriority() {
  return SetContextPriority(context_priority_);
}

Ort::Status QnnBackendManager::CreateContext(bool enable_htp_weight_sharing) {
  if (true == context_created_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, "Context created already.");
    return Ort::Status();
  }

  QnnContext_Config_t context_config_weight_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t custom_config;
  custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  custom_config.weightSharingEnabled = enable_htp_weight_sharing;
  context_config_weight_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_weight_sharing.customConfig = &custom_config;

  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, context_priority_config));

  const QnnContext_Config_t* npu_context_configs[] = {&context_priority_config,
                                                      &context_config_weight_sharing,
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

  RETURN_IF(QNN_CONTEXT_NO_ERROR != result,
            ("Failed to create context. Error: " + QnnErrorHandleToString(result)).c_str());

  RETURN_IF_ERROR(AddQnnContextHandle(context));

  context_created_ = true;
  return Ort::Status();
}

Ort::Status QnnBackendManager::ReleaseContext() {
  if (false == context_created_) {
    return Ort::Status();
  }

  // release QNN context handles
  contexts_.clear();
  context_map_.clear();

  context_created_ = false;
  return Ort::Status();
}

std::unique_ptr<unsigned char[]> QnnBackendManager::GetContextBinaryBuffer(uint64_t& written_buffer_size) {
  if (nullptr == qnn_interface_.contextGetBinarySize ||
      nullptr == qnn_interface_.contextGetBinary) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Failed to get valid function pointer.");
    return nullptr;
  }
  if (contexts_.size() <= 0) {
    ORT_CXX_API_THROW("No valid QNN context!", ORT_EP_FAIL);
  }
  uint64_t required_buffer_size(0);
  // Generate all graphs in one single context
  Qnn_ErrorHandle_t rt = qnn_interface_.contextGetBinarySize(contexts_[0], &required_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                ("Failed to get QNN context binary size. Error: " + QnnErrorHandleToString(rt)).c_str());
    return nullptr;
  }

  std::unique_ptr<unsigned char[]> context_buffer = std::make_unique<unsigned char[]>(required_buffer_size);
  if (nullptr == context_buffer) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Failed to allocate buffer for context cache.");
    return nullptr;
  }

  rt = qnn_interface_.contextGetBinary(contexts_[0],
                                       reinterpret_cast<void*>(context_buffer.get()),
                                       required_buffer_size,
                                       &written_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                ("Failed to get context binary. Error: " + QnnErrorHandleToString(rt)).c_str());
    return nullptr;
  }

  if (required_buffer_size < written_buffer_size) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                ("Context written buffer size: " + std::to_string(written_buffer_size) +
                 " exceeds allocated buffer size: " + std::to_string(required_buffer_size))
                    .c_str());
    return nullptr;
  }

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Get context binary buffer succeed.");
  return context_buffer;
}

Ort::Status QnnBackendManager::GetMaxSpillFillBufferSize(unsigned char* buffer,
                                                         uint64_t buffer_length,
                                                         uint64_t& max_spill_fill_buffer_size) {
  max_spill_fill_buffer_size = 0;
  // spill fill starts from 2.28
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)
  bool result = nullptr == qnn_sys_interface_.systemContextCreate ||
                nullptr == qnn_sys_interface_.systemContextGetBinaryInfo ||
                nullptr == qnn_sys_interface_.systemContextFree;
  RETURN_IF(result, "Failed to get valid function pointer.");

  QnnSystemContext_Handle_t sys_ctx_handle = nullptr;
  auto rt = qnn_sys_interface_.systemContextCreate(&sys_ctx_handle);
  RETURN_IF(QNN_SUCCESS != rt, "Failed to create system handle.");

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size{0};
  rt = qnn_sys_interface_.systemContextGetBinaryInfo(sys_ctx_handle,
                                                     static_cast<void*>(buffer),
                                                     buffer_length,
                                                     &binary_info,
                                                     &binary_info_size);
  RETURN_IF(QNN_SUCCESS != rt, "Failed to get context binary info.");

  // binary_info life cycle is here
  // Binary info to graph info
  // retrieve Qnn graph info from binary info
  RETURN_IF(nullptr == binary_info, "Qnn cached binary info is nullptr.");
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
    return MAKE_EP_FAIL("Unsupported context binary info version.");
  }

  for (uint32_t i = 0; i < graph_count; ++i) {
    if (graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      auto htp_graph_info = reinterpret_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(graphs_info[i].graphInfoV3.graphBlobInfo);
      if (htp_graph_info->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
        auto spill_fill_buffer_size = htp_graph_info->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
        max_spill_fill_buffer_size = spill_fill_buffer_size > max_spill_fill_buffer_size ? spill_fill_buffer_size : max_spill_fill_buffer_size;
      } else {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Unknown context binary graph info blob version.");
      }
    } else if (graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 ||
               graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  "Skip retrieve spill file buffer size, it is not supported with graph info v1 & v2.");
    } else {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Unknown context binary graph info version.");
    }
  }
#else
  ORT_UNUSED_PARAMETER(buffer);
  ORT_UNUSED_PARAMETER(buffer_length);
#endif

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Get max spill fill buffer size completed.");
  return Ort::Status();
}

Ort::Status QnnBackendManager::LoadCachedQnnContextFromBuffer(
    char* buffer,
    uint64_t buffer_length,
    std::string node_name,
    std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models,
    int64_t max_spill_fill_size) {
  bool result = nullptr == qnn_sys_interface_.systemContextCreate ||
                nullptr == qnn_sys_interface_.systemContextGetBinaryInfo ||
                nullptr == qnn_sys_interface_.systemContextFree;
  RETURN_IF(result, "Failed to get valid function pointer.");

  QnnSystemContext_Handle_t sys_ctx_handle = nullptr;
  auto rt = qnn_sys_interface_.systemContextCreate(&sys_ctx_handle);
  RETURN_IF(QNN_SUCCESS != rt, "Failed to create system handle.");

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size{0};
  rt = qnn_sys_interface_.systemContextGetBinaryInfo(sys_ctx_handle,
                                                     static_cast<void*>(buffer),
                                                     buffer_length,
                                                     &binary_info,
                                                     &binary_info_size);
  RETURN_IF(QNN_SUCCESS != rt, "Failed to get context binary info.");

  // binary_info life cycle is here
  // Binary info to graph info
  // retrieve Qnn graph info from binary info
  RETURN_IF(nullptr == binary_info, "Qnn cached binary info is nullptr.");
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
    return MAKE_EP_FAIL("Unsupported context binary info version.");
  }

  RETURN_IF(graph_count < 1 || graphs_info == nullptr, "Failed to get graph info from Qnn cached context.");
  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("Graph count from QNN context: " + std::to_string(graph_count)).c_str());

  Qnn_ContextHandle_t context = nullptr;
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  if (vtcm_backup_buffer_sharing_enabled_) {
    if (ep_context_handle_map_.find(node_name) != ep_context_handle_map_.end()) {
      context = ep_context_handle_map_.at(node_name);
    }
    RETURN_IF(nullptr == context, ("Failed to retrieve context for " + node_name).c_str());

  } else {
#endif
    QnnContext_Config_t qnn_context_config = QNN_CONTEXT_CONFIG_INIT;
    RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, qnn_context_config));

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
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Max spill fill buffer size: " + std::to_string(max_spill_fill_size)).c_str());

    const QnnContext_Config_t* context_configs[] = {&qnn_context_config, spill_fill_config_pointer, nullptr};

    RETURN_IF(nullptr == qnn_interface_.contextCreateFromBinary,
              "Invalid function pointer for contextCreateFromBinary.");

    qnn::profile::ProfilingInfo profiling_info;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (ProfilingEnabled()) {
      profiling_info.start_time = qnn::utils::GetTimeStampInUs();
    }
#endif

    rt = qnn_interface_.contextCreateFromBinary(backend_handle_,
                                                device_handle_,
                                                context_configs,
                                                static_cast<void*>(buffer),
                                                buffer_length,
                                                &context,
                                                profile_backend_handle_);

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (ProfilingEnabled()) {
      profiling_info.stop_time = qnn::utils::GetTimeStampInUs();
      profiling_info.method_type = ProfilingMethodType::CREATE_FROM_BINARY;
      profiling_info.graph_name = node_name;
    }
#endif

    RETURN_IF(QNN_SUCCESS != rt,
              ("Failed to create context from binary. Error: " + QnnErrorHandleToString(rt)).c_str());
    RETURN_IF_ERROR(AddQnnContextHandle(context));

    RETURN_IF_ERROR(ExtractBackendProfilingInfo(profiling_info));

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  }
#endif

  if (1 == graph_count) {
    // in case the EPContext node is generated from script
    // the graph name from the context binary may not match the EPContext node name
    auto qnn_model = std::make_unique<qnn::QnnModel>(this, api_ptrs_);
    RETURN_IF_ERROR(qnn_model->DeserializeGraphInfoFromBinaryInfo(graphs_info[0], context));
    qnn_models.emplace(node_name, std::move(qnn_model));
  } else {
    for (uint32_t i = 0; i < graph_count; ++i) {
      auto qnn_model = std::make_unique<qnn::QnnModel>(this, api_ptrs_);
      RETURN_IF_ERROR(qnn_model->DeserializeGraphInfoFromBinaryInfo(graphs_info[i], context));
      qnn_models.emplace(graphs_info[i].graphInfoV1.graphName, std::move(qnn_model));
    }
  }

  qnn_sys_interface_.systemContextFree(sys_ctx_handle);
  sys_ctx_handle = nullptr;
  context_created_ = true;

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Load from cached QNN Context completed.");
  return Ort::Status();
}

// need to load system lib if load from Qnn context binary
// or generate Qnn context binary is enabled -- to get the max spill fill buffer size
Ort::Status QnnBackendManager::SetupBackend(
    bool load_from_cached_context,
    bool need_load_system_lib,
    bool share_ep_contexts,
    bool enable_vtcm_backup_buffer_sharing,
    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>& context_bin_map) {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (backend_setup_completed_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Backend setup already!");

    if (vtcm_backup_buffer_sharing_enabled_) {
      // If a context bin filepath has not been processed yet,
      // then a new context must be created for the set of context bins
      auto first_mapping_it = ep_context_handle_map_.find(context_bin_map.begin()->first);
      if (first_mapping_it == ep_context_handle_map_.end()) {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Creating context for new set of context binaries");
        return CreateContextVtcmBackupBufferSharingEnabled(context_bin_map);
      }

      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Mapping contexts to new EP main context nodes");

      for (auto& it : context_bin_map) {
        auto context_bin_filepath = it.first;
        auto ep_node_names = *(it.second);

        auto context = ep_context_handle_map_.at(context_bin_filepath);
        for (auto node_name : ep_node_names) {
          ep_context_handle_map_.emplace(node_name, context);
        }
      }
    }
    return Ort::Status();
  }

  vtcm_backup_buffer_sharing_enabled_ = enable_vtcm_backup_buffer_sharing;

  auto status = Ort::Status();
  if (!qnn_serializer_config_) {
    status = LoadBackend();
  } else {
    status = LoadQnnSerializerBackend();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "LoadBackend succeed.");
  }

  if (status.IsOK() && (load_from_cached_context || need_load_system_lib)) {
    status = LoadQnnSystemLib();
  }

  if (status.IsOK()) {
    sdk_build_version_ = GetBackendBuildId();
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Backend build version: " + sdk_build_version_).c_str());
  }

  if (status.IsOK()) {
    status = InitializeQnnLog();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "SetLogger succeed.");
  }

  if (status.IsOK()) {
    status = InitializeBackend();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "InitializeBackend succeed.");
  }

  if (status.IsOK()) {
    status = CreateDevice();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "CreateDevice succeed.");
  }

  if (status.IsOK()) {
    status = InitializeProfiling();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "InitializeProfiling succeed.");
  }

  if (status.IsOK()) {
    RETURN_IF_ERROR(LoadOpPackage());
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "LoadOpPackage succeed.");
  }

  bool enable_htp_weight_sharing = false;
  if (share_ep_contexts && !load_from_cached_context) {
#if defined(__aarch64__) || defined(_M_ARM64)
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                "Weight sharing only available with offline generation on x64 platform, not work on real device.");
#else
    enable_htp_weight_sharing = true;
#endif
  }

  if (status.IsOK() && (vtcm_backup_buffer_sharing_enabled_ || !load_from_cached_context)) {
    status = vtcm_backup_buffer_sharing_enabled_ ? CreateContextVtcmBackupBufferSharingEnabled(context_bin_map)
                                                 : CreateContext(enable_htp_weight_sharing);

    if (status.IsOK()) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "CreateContext succeed.");
    }
  }

  if (status.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "QNN SetupBackend succeed");
    backend_setup_completed_ = true;
  } else {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Failed to setup so cleaning up");
    ReleaseResources();
  }

  return status;
}

Ort::Status QnnBackendManager::CreateHtpPowerCfgId(uint32_t device_id,
                                                   uint32_t core_id,
                                                   uint32_t& htp_power_config_id) {
  // This function is called in QNN EP's OnRunStart() even if QNN backend setup failed and the model is assigned
  // to a different EP. Therefore, we have to check that backend setup actually completed before trying to
  // create an HTP power config ID. Otherwise, this causes a segfault because the QNN backend lib is unloaded.
  RETURN_IF_NOT(backend_setup_completed_, "Cannot create HTP power config ID if backend setup is not complete.");
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
            ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;
  // Get power client id
  status = htp_perf_infra.createPowerConfigId(device_id, core_id, &htp_power_config_id);
  RETURN_IF(QNN_SUCCESS != status, "createPowerConfigId failed.");

  return Ort::Status();
}

Ort::Status QnnBackendManager::SetHtpPowerConfig(uint32_t htp_power_config_client_id,
                                                 HtpPerformanceMode htp_performance_mode) {
  // This function is called in QNN EP's OnRunStart() even if QNN backend setup failed and the model is assigned
  // to a different EP. Therefore, we have to check that backend setup actually completed before trying to
  // set an HTP power config ID. Otherwise, this causes a segfault because the QNN backend lib is unloaded.
  RETURN_IF_NOT(backend_setup_completed_, "Cannot set HTP power config ID if backend setup is not complete.");
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
            ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

  constexpr const int kNumConfigs = 1;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs(
      kNumConfigs);
  QnnHtpPerfInfrastructure_PowerConfig_t& dcvs_config = power_configs[0];
  dcvs_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = dcvs_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  // choose performance mode
  switch (htp_performance_mode) {
    case HtpPerformanceMode::kHtpBurst:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMinLatency;
      dcvs_v3.dcvsEnable = kDcvsDisable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      break;
    case HtpPerformanceMode::kHtpSustainedHighPerformance:
    case HtpPerformanceMode::kHtpHighPerformance:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepLowLatency;
      dcvs_v3.dcvsEnable = kDcvsDisable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
      break;
    case HtpPerformanceMode::kHtpBalanced:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      break;
    case HtpPerformanceMode::kHtpLowBalanced:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
      break;
    case HtpPerformanceMode::kHtpHighPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      break;
    case HtpPerformanceMode::kHtpPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
      break;
    case HtpPerformanceMode::kHtpLowPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
      break;
    case HtpPerformanceMode::kHtpExtremePowerSaver:
      dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
      break;
    default:
      ORT_CXX_API_THROW(("Invalid performance profile " +
                         std::to_string(static_cast<uint8_t>(htp_performance_mode)))
                            .c_str(),
                        ORT_EP_FAIL);
      break;
  }
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr = ObtainNullTermPtrVector(power_configs);
  status = htp_perf_infra.setPowerConfig(htp_power_config_client_id, perf_power_configs_ptr.data());
  RETURN_IF(QNN_SUCCESS != status, "setPowerConfig failed for HTP performance mode.");

  return Ort::Status();
}

Ort::Status QnnBackendManager::SetRpcPowerConfigs(uint32_t htp_power_config_client_id,
                                                  uint32_t rpc_control_latency,
                                                  uint32_t rpc_polling_time) {
  // This function is called in QNN EP's OnRunStart() even if QNN backend setup failed and the model is assigned
  // to a different EP. Therefore, we have to check that backend setup actually completed before trying to
  // set RPC control latency. Otherwise, this causes a segfault because the QNN backend library is unloaded.
  RETURN_IF_NOT(backend_setup_completed_, "Cannot set HTP RPC control latency if backend setup is not complete.");

  constexpr int kNumRpcPollingPowerConfigs = 2;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> rpc_power_configs;
  rpc_power_configs.reserve(kNumRpcPollingPowerConfigs);

  // Set rpc control latency here
  if (rpc_control_latency != 0) {
    auto& rpc_control_latency_cfg = rpc_power_configs.emplace_back();
    rpc_control_latency_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc_control_latency_cfg.rpcControlLatencyConfig = rpc_control_latency;
  }

  // Note: v68 does not support rpc polling mode
  if (rpc_polling_time != 0) {
    auto& rpc_polling_time_cfg = rpc_power_configs.emplace_back();
    rpc_polling_time_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpc_polling_time_cfg.rpcPollingTimeConfig = rpc_polling_time;
  }

  if (rpc_power_configs.size() > 0) {
    QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
    auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
    RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

    auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
    RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
              ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
    QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

    std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr =
        ObtainNullTermPtrVector(rpc_power_configs);
    status = htp_perf_infra.setPowerConfig(htp_power_config_client_id, perf_power_configs_ptr.data());
    RETURN_IF(QNN_SUCCESS != status, "setPowerConfig failed for RPC control latency.");
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::DestroyHTPPowerConfigID(uint32_t htp_power_config_id) {
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
            ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

  Qnn_ErrorHandle_t destroy_ret = htp_perf_infra.destroyPowerConfigId(htp_power_config_id);
  RETURN_IF(QNN_SUCCESS != destroy_ret, "destroyPowerConfigId failed.");
  return Ort::Status();
}

Ort::Status QnnBackendManager::TerminateQnnLog() {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);

  if (nullptr != qnn_interface_.logFree && nullptr != log_handle_) {
    auto ret_val = qnn_interface_.logFree(log_handle_);

    // Reset QNN log handle to nullptr so other threads that are waiting on logger_recursive_mutex_ know it was freed.
    log_handle_ = nullptr;
    RETURN_IF(QNN_SUCCESS != ret_val, "Unable to terminate logging in the backend.");
  }

  return Ort::Status();
}

void QnnBackendManager::ReleaseResources() {
  auto result = ReleaseContext();
  if (!result.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ReleaseContext: " + result.GetErrorMessage()).c_str());
  }

  result = ReleaseProfilehandle();
  if (!result.IsOK()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                ("Failed to ReleaseProfilehandle: " + result.GetErrorMessage()).c_str());
  }

  result = ReleaseDevice();
  if (!result.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ReleaseDevice: " + result.GetErrorMessage()).c_str());
  }

  result = ShutdownBackend();
  if (!result.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ShutdownBackend: " + result.GetErrorMessage()).c_str());
  }

  result = TerminateQnnLog();
  if (!result.IsOK()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, ("Failed to TerminateQnnLog: " + result.GetErrorMessage()).c_str());
  }

  if (backend_lib_handle_) {
    result = UnloadLib(backend_lib_handle_);
    if (!result.IsOK()) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_ERROR,
                  ("Failed to unload backend library: " + result.GetErrorMessage()).c_str());
    }
  }

  backend_setup_completed_ = false;

  return;
}

Ort::Status QnnBackendManager::ExtractBackendProfilingInfo(qnn::profile::ProfilingInfo& profiling_info) {
  if (ProfilingLevel::OFF == profiling_level_merge_ || ProfilingLevel::INVALID == profiling_level_merge_) {
    return Ort::Status();
  }

  bool tracelogging_provider_ep_enabled = false;
#ifdef _WIN32
  auto& provider = QnnTelemetry::Instance();
  if (provider.IsEnabled()) {
    auto level = provider.Level();
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(qnn::ORTTraceLoggingKeyword::Profiling)) != 0 && level >= 5) {
      tracelogging_provider_ep_enabled = true;
    }
  }
#endif  // defined(_WIN32)

  // ETW disabled previously, but enabled now
  if (ProfilingLevel::INVALID == profiling_level_etw_ && tracelogging_provider_ep_enabled) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                "ETW disabled previously, but enabled now. Can't do the switch! Won't output any profiling.");
    return Ort::Status();
  }

  // ETW enabled previously, but disabled now
  if (ProfilingLevel::INVALID != profiling_level_etw_ && !tracelogging_provider_ep_enabled) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                "ETW enabled previously, but disabled now. Can't do the switch! Won't output any profiling.");
    return Ort::Status();
  }

  RETURN_IF(!tracelogging_provider_ep_enabled && profiling_file_path_.empty(),
            "Need to specify a CSV file via provider option profiling_file_path if ETW not enabled.");

  RETURN_IF(nullptr == profile_backend_handle_, "Backend profile handle not valid.");

  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("Extracting profiling events for graph " + profiling_info.graph_name).c_str());

  const QnnProfile_EventId_t* profile_events{nullptr};
  uint32_t num_events{0};
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetEvents(profile_backend_handle_, &profile_events, &num_events);
  if (qnn_serializer_config_) {  // Using QNN Saver or IR backend
    // QNN SDK 2.28.2 returns QNN_SAVER_ERROR_DUMMY_RETVALUE, but previous QNN versions return QNN_PROFILE_NO_ERROR.
    // We accept both values.
    RETURN_IF(QNN_PROFILE_NO_ERROR != result && QNN_SAVER_ERROR_DUMMY_RETVALUE != result,
              ("Failed to get profile events. Error: " + QnnErrorHandleToString(result)).c_str());
  } else {
    RETURN_IF(QNN_PROFILE_NO_ERROR != result,
              ("Failed to get profile events. Error: " + QnnErrorHandleToString(result)).c_str());
  }

  if (num_events > 0) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("profile_events: " + std::to_string(*profile_events) +
                 " num_events: " + std::to_string(num_events))
                    .c_str());

    bool backendSupportsExtendedEventData = false;
    Qnn_ErrorHandle_t resultPropertyHasCapability =
        qnn_interface_.propertyHasCapability(QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT);
    uint16_t errorCodePropertyHasCapability = static_cast<uint16_t>(resultPropertyHasCapability & 0xFFFF);
    if (errorCodePropertyHasCapability == QNN_PROPERTY_SUPPORTED) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "The QNN backend supports extended event data.");
      backendSupportsExtendedEventData = true;
    } else {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "The QNN backend does not support extended event data.");
    }

    profiling_info.csv_output_filepath = profiling_file_path_;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    profiling_info.num_events = num_events;
#endif

    profile::Serializer profile_writer(profiling_info,
                                       qnn_sys_interface_,
                                       tracelogging_provider_ep_enabled);
    if (!profiling_file_path_.empty()) {
      RETURN_IF_ERROR(profile_writer.InitCsvFile());
    }

    for (size_t event_idx = 0; event_idx < num_events; event_idx++) {
      RETURN_IF_ERROR(ExtractProfilingEvent(*(profile_events + event_idx),
                                            "ROOT",
                                            profile_writer,
                                            backendSupportsExtendedEventData));
      RETURN_IF_ERROR(ExtractProfilingSubEvents(*(profile_events + event_idx),
                                                profile_writer,
                                                backendSupportsExtendedEventData));
    }

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    RETURN_IF_ERROR(profile_writer.SerializeEventsToQnnLog());
#endif

    if (!profiling_file_path_.empty()) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Wrote QNN profiling events (" + std::to_string(num_events) +
                   ") to file (" + profiling_file_path_ + ")")
                      .c_str());
    }

    if (tracelogging_provider_ep_enabled) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Wrote QNN profiling events (" + std::to_string(num_events) + ") to ETW").c_str());
    }
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id,
                                                         profile::Serializer& profile_writer,
                                                         bool useExtendedEventData) {
  const QnnProfile_EventId_t* profile_sub_events{nullptr};
  uint32_t num_sub_events{0};
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
  RETURN_IF(QNN_PROFILE_NO_ERROR != result,
            ("Failed to get profile sub events. Error: " + QnnErrorHandleToString(result)).c_str());

  if (num_sub_events > 0) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("profile_sub_events: " + std::to_string(*profile_sub_events) +
                 " num_sub_events: " + std::to_string(num_sub_events))
                    .c_str());

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
      RETURN_IF_ERROR(profile_writer.SetParentSystemEvent(subevent_id, parent_system_event));
#endif
      RETURN_IF_ERROR(ExtractProfilingEvent(subevent_id, "SUB-EVENT", profile_writer, useExtendedEventData));
      RETURN_IF_ERROR(ExtractProfilingSubEvents(subevent_id, profile_writer, useExtendedEventData));
    }

    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Wrote QNN profiling sub events (" + std::to_string(num_sub_events) + ")").c_str());
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id,
                                                     const std::string& event_level,
                                                     profile::Serializer& profile_writer,
                                                     bool useExtendedEventData) {
  if (useExtendedEventData) {
    return ExtractProfilingEventExtended(profile_event_id, event_level, profile_writer);
  } else {
    return ExtractProfilingEventBasic(profile_event_id, event_level, profile_writer);
  }
}

Ort::Status QnnBackendManager::ExtractProfilingEventBasic(QnnProfile_EventId_t profile_event_id,
                                                          const std::string& event_level,
                                                          profile::Serializer& profile_writer) {
  QnnProfile_EventData_t event_data;
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetEventData(profile_event_id, &event_data);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(result & 0xFFFF);
  RETURN_IF(QNN_PROFILE_NO_ERROR != result,
            ("Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode))).c_str());

  RETURN_IF_ERROR(profile_writer.ProcessEvent(profile_event_id, event_level, event_data));

  return Ort::Status();
}

Ort::Status QnnBackendManager::ExtractProfilingEventExtended(QnnProfile_EventId_t profile_event_id,
                                                             const std::string& event_level,
                                                             profile::Serializer& profile_writer) {
  QnnProfile_ExtendedEventData_t event_data_extended;
  auto resultGetExtendedEventData = qnn_interface_.profileGetExtendedEventData(profile_event_id, &event_data_extended);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(resultGetExtendedEventData & 0xFFFF);
  RETURN_IF(QNN_PROFILE_NO_ERROR != errorCode,
            ("Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode))).c_str());

  RETURN_IF_ERROR(profile_writer.ProcessExtendedEvent(profile_event_id, event_level, event_data_extended));

  return Ort::Status();
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
    auto pathstring = std::filesystem::path(OrtGetRuntimePath()) / file_path;
    auto absolute_path = pathstring.c_str();
    if (std::filesystem::exists(pathstring)) {
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

Ort::Status QnnBackendManager::UnloadLib(void* handle) {
  if (!handle) {
    return Ort::Status();
  }

#ifdef _WIN32
  HMODULE mod = static_cast<HMODULE>(handle);

  if (FreeLibrary(mod) == 0) {
    return MAKE_EP_FAIL("Failed to free library.");
  }
  mod_handles_.erase(mod);
#else
  auto rt = ::dlclose(handle);
  if (rt != 0) {
    return MAKE_EP_FAIL("Failed to free library.");
  }
#endif  // defined(_WIN32)

  return Ort::Status();
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

Ort::Status QnnBackendManager::AddQnnContextHandle(Qnn_ContextHandle_t raw_context_handle) {
  auto free_context_handle = [this](Qnn_ContextHandle_t raw_context_handle) {
    const auto free_result = qnn_interface_.contextFree(raw_context_handle, nullptr);
    if (free_result != QNN_CONTEXT_NO_ERROR) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_ERROR,
                  ("qnn_interface.contextFree() failed: " + utils::GetVerboseQnnErrorMessage(qnn_interface_, free_result)).c_str());
    }
  };

  // take ownership of `raw_context_handle`
  auto context_handle = UniqueQnnContextHandle(raw_context_handle, free_context_handle);
  auto mem_handle_manager = std::make_unique<QnnContextMemHandleManager>(GetQnnInterface(), raw_context_handle,
                                                                         logger_);

  auto context_handle_record = std::make_shared<QnnContextHandleRecord>();
  context_handle_record->context_handle = std::move(context_handle);
  context_handle_record->mem_handles = std::move(mem_handle_manager);

  const bool inserted = context_map_.try_emplace(raw_context_handle, std::move(context_handle_record)).second;
  RETURN_IF_NOT(inserted, "QNN context was already added.");

  contexts_.push_back(raw_context_handle);

  return Ort::Status();
}

Ort::Status QnnBackendManager::GetOrRegisterContextMemHandle(Qnn_ContextHandle_t context_handle,
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
  RETURN_IF_NOT(context_handle_record_it != context_map_.end(), "QNN context not found.");

  auto& context_handle_record = context_handle_record_it->second;
  auto& context_mem_handle_manager = context_handle_record->mem_handles;

  bool did_register{};
  RETURN_IF_ERROR(context_mem_handle_manager->GetOrRegister(shared_memory_address, qnn_tensor,
                                                            mem_handle, did_register));

  if (did_register) {
    HtpSharedMemoryAllocator::AllocationCleanUpFn unregister_mem_handle =
        [&logger = logger_,
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
            std::ostringstream oss;
            oss << "Failed to unregister shared memory mem handle for address: "
                << shared_memory_address
                << ", error: "
                << unregister_status.GetErrorMessage();
            ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
          }
        };

    RETURN_IF_ERROR(HtpSharedMemoryAllocator::AddAllocationCleanUp(shared_memory_address,
                                                                   std::move(unregister_mem_handle)));
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
