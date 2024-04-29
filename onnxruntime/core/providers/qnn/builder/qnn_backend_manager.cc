// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_backend_manager.h"
#include "qnn_model.h"
#include <filesystem>
#include <fstream>
#include <string>
#include "QnnOpDef.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "CPU/QnnCpuCommon.h"
// TODO: not exist for Windows yet
// #include "GPU/QnnGpuCommon.h"
#include "DSP/QnnDspCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "core/common/gsl.h"
#include "core/framework/endian_utils.h"
#include "core/common/logging/capture.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/qnn_configs_helper.h"

#ifdef _WIN32
#include <winmeta.h>
#include "core/platform/tracing.h"
#endif

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

template <typename F, class T>
Status QnnBackendManager::GetQnnInterfaceProvider(const char* lib_path,
                                                  const char* interface_provider_name,
                                                  void** backend_lib_handle,
                                                  Qnn_Version_t req_version,
                                                  T** interface_provider) {
  std::string error_msg;
  *backend_lib_handle = LoadLib(lib_path,
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_LOCAL),
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
      // TODO: update once it's ready for Widows
      // case QNN_BACKEND_ID_GPU:
      //  qnn_backend_type_ = QnnBackendType::GPU;
      //  break;
    case QNN_BACKEND_ID_DSP:
      qnn_backend_type_ = QnnBackendType::DSP;
      break;
    case QNN_BACKEND_ID_HTP:
      qnn_backend_type_ = QnnBackendType::HTP;
      break;
    default:
      qnn_backend_type_ = QnnBackendType::CPU;
      break;
  }
}

Status QnnBackendManager::LoadBackend() {
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

// Loads the intended backend (e.g., HTP, CPU, etc) to get its type, and then
// sets QNN Saver as the active backend. QNN op builders will still see the intended backend (e.g., HTP)
// as the backend type to ensure they emit the expected QNN API calls.
//
// QNN Saver is a "debugging" backend that serializes all QNN API calls (and weights) into local files.
// This information can be used to debug issues by replaying QNN API calls with another backend.
Status QnnBackendManager::LoadQnnSaverBackend() {
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

  // Load the QNN Saver backend and set it as the activate backend.
  QnnInterface_t* saver_interface_provider{nullptr};
  auto saver_rt = GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t,
                                          QnnInterface_t>(qnn_saver_path_.c_str(),
                                                          "QnnInterface_getProviders",
                                                          &backend_lib_handle_,  // NOTE: QNN Saver library handle is set
                                                          {QNN_API_VERSION_MAJOR,
                                                           QNN_API_VERSION_MINOR,
                                                           QNN_API_VERSION_PATCH},
                                                          &saver_interface_provider);
  ORT_RETURN_IF_ERROR(saver_rt);
  qnn_interface_ = saver_interface_provider->QNN_INTERFACE_VER_NAME;  // NOTE: QNN Saver will provide the interfaces

  Qnn_Version_t backend_interface_version = GetQnnInterfaceApiVersion(backend_interface_provider);
  Qnn_Version_t saver_interface_version = GetQnnInterfaceApiVersion(saver_interface_provider);

  LOGS_DEFAULT(INFO) << "Using QNN Saver version: " << saver_interface_version.major << "."
                     << saver_interface_version.minor << "." << saver_interface_version.patch
                     << " provider name : " << saver_interface_provider->providerName;

  LOGS_DEFAULT(INFO) << "Intended backend provider name: " << backend_interface_provider->providerName
                     << " backend id: " << backend_id
                     << " interface version: " << backend_interface_version.major
                     << "." << backend_interface_version.minor << "." << backend_interface_version.patch;

  return Status::OK();
}

Status QnnBackendManager::LoadQnnSystemLib() {
#ifdef _WIN32
  std::string system_lib_file = "QnnSystem.dll";
#else
  std::string system_lib_file = "libQnnSystem.so";
#endif  // #ifdef _WIN32
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

  return Status::OK();
}

void QnnLogging(const char* format,
                QnnLog_Level_t level,
                uint64_t timestamp,
                va_list argument_parameter) {
  ORT_UNUSED_PARAMETER(level);
  ORT_UNUSED_PARAMETER(timestamp);

  // Always output Qnn log as Ort verbose log
  const auto& logger = ::onnxruntime::logging::LoggingManager::DefaultLogger();
  const auto severity = ::onnxruntime::logging::Severity::kVERBOSE;
  const auto data_type = ::onnxruntime::logging::DataType::SYSTEM;
  if (logger.OutputIsEnabled(severity, data_type)) {
    ::onnxruntime::logging::Capture(logger,
                                    severity,
                                    ::onnxruntime::logging::Category::onnxruntime,
                                    data_type,
                                    ORT_WHERE)
        .ProcessPrintf(format, argument_parameter);
  }
}

void QnnBackendManager::InitializeQnnLog() {
  // Set Qnn log level align with Ort log level
  QnnLog_Level_t qnn_log_level = QNN_LOG_LEVEL_WARN;
  auto ort_log_level = logger_->GetSeverity();
  switch (ort_log_level) {
    case logging::Severity::kVERBOSE:
      qnn_log_level = QNN_LOG_LEVEL_DEBUG;
      break;
    case logging::Severity::kINFO:
      qnn_log_level = QNN_LOG_LEVEL_INFO;
      break;
    case logging::Severity::kWARNING:
      qnn_log_level = QNN_LOG_LEVEL_WARN;
      break;
    case logging::Severity::kERROR:
      qnn_log_level = QNN_LOG_LEVEL_ERROR;
      break;
    default:
      break;
  }
  LOGS(*logger_, VERBOSE) << "Set Qnn log level: " << qnn_log_level;

  if (QNN_SUCCESS != qnn_interface_.logCreate(QnnLogging, qnn_log_level, &log_handle_)) {
    LOGS(*logger_, WARNING) << "Unable to initialize logging in the QNN backend.";
  }
}

Status QnnBackendManager::InitializeBackend() {
  if (true == backend_initialized_) {
    LOGS_DEFAULT(INFO) << "Backend initialized already.";
    return Status::OK();
  }

  auto result = qnn_interface_.backendCreate(log_handle_, (const QnnBackend_Config_t**)backend_config_, &backend_handle_);
  ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != result, "Failed to initialize backend");

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
      QnnHtpDevice_CustomConfig_t& custom_config = device_configs_builder.PushCustomConfig();
      custom_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
      custom_config.socModel = soc_model_;

      QnnDevice_Config_t& device_config = device_configs_builder.PushConfig();
      device_config.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      device_config.customConfig = &custom_config;
    }

    // Set the minimum HTP architecture. The driver will use ops that are compatible with this minimum architecture.
    if (htp_arch_ != QNN_HTP_DEVICE_ARCH_NONE) {
      QnnHtpDevice_CustomConfig_t& custom_config = device_configs_builder.PushCustomConfig();
      custom_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
      custom_config.arch.arch = htp_arch_;
      custom_config.arch.deviceId = device_id_;

      QnnDevice_Config_t& device_config = device_configs_builder.PushConfig();
      device_config.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      device_config.customConfig = &custom_config;
    }
  }

  LOGS_DEFAULT(INFO) << "Create device.";
  if (nullptr != qnn_interface_.deviceCreate) {
    auto result = qnn_interface_.deviceCreate(log_handle_, device_configs_builder.GetQnnConfigs(), &device_handle_);
    if (QNN_SUCCESS != result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create device. Error: ", result);
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
    auto result = qnn_interface_.deviceFree(device_handle_);
    if (QNN_SUCCESS != result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to release device. Error: ", result);
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

  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  if (ProfilingLevel::BASIC == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  } else if (ProfilingLevel::DETAILED == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
  }
  auto result = qnn_interface_.profileCreate(backend_handle_, qnn_profile_level, &profile_backend_handle_);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to create QNN profile!");

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

Status QnnBackendManager::CreateContext() {
  if (true == context_created_) {
    LOGS_DEFAULT(INFO) << "Context created already.";
    return Status::OK();
  }

  QnnContext_Config_t qnn_context_config = QNN_CONTEXT_CONFIG_INIT;
  ORT_RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, qnn_context_config));
  const QnnContext_Config_t* context_configs[] = {&qnn_context_config, nullptr};

  auto result = qnn_interface_.contextCreate(backend_handle_,
                                             device_handle_,
                                             context_configs,
                                             &context_);

  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to create context.");

  context_created_ = true;
  return Status::OK();
}

Status QnnBackendManager::ReleaseContext() {
  if (false == context_created_) {
    return Status::OK();
  }

  auto result = qnn_interface_.contextFree(context_, nullptr);
  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to release context.");

  context_created_ = false;
  return Status::OK();
}

std::unique_ptr<unsigned char[]> QnnBackendManager::GetContextBinaryBuffer(uint64_t& written_buffer_size) {
  if (nullptr == qnn_interface_.contextGetBinarySize ||
      nullptr == qnn_interface_.contextGetBinary) {
    LOGS(*logger_, ERROR) << "Failed to get valid function pointer.";
    return nullptr;
  }

  uint64_t required_buffer_size(0);
  Qnn_ErrorHandle_t rt = qnn_interface_.contextGetBinarySize(context_, &required_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    LOGS(*logger_, ERROR) << "Failed to get QNN context binary size. Error code: " << rt;
    return nullptr;
  }

  std::unique_ptr<unsigned char[]> context_buffer = std::make_unique<unsigned char[]>(required_buffer_size);
  if (nullptr == context_buffer) {
    LOGS(*logger_, ERROR) << "Failed to allocate buffer for context cache.";
    return nullptr;
  }

  rt = qnn_interface_.contextGetBinary(context_,
                                       reinterpret_cast<void*>(context_buffer.get()),
                                       required_buffer_size,
                                       &written_buffer_size);
  if (QNN_CONTEXT_NO_ERROR != rt) {
    LOGS(*logger_, ERROR) << "Failed to get context binary.";
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

Status QnnBackendManager::LoadCachedQnnContextFromBuffer(char* buffer, uint64_t buffer_length,
                                                         std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models) {
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
  // retrieve Qnn graph infor from binary info
  ORT_RETURN_IF(nullptr == binary_info, "Qnn cached binary info is nullptr.");
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graph_count = binary_info->contextBinaryInfoV1.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV1.graphs;
  } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graph_count = binary_info->contextBinaryInfoV2.numGraphs;
    graphs_info = binary_info->contextBinaryInfoV2.graphs;
  }

  ORT_RETURN_IF(graph_count < 1 || graphs_info == nullptr, "Failed to get graph info from Qnn cached context.");
  LOGS(*logger_, VERBOSE) << "Graph count from QNN context: " << graph_count << ", EPContext node count: " << qnn_models.size();
  ORT_RETURN_IF(graph_count != qnn_models.size(), "Graph count from QNN context not equal to EPContext node count.");

  ORT_RETURN_IF(nullptr == qnn_interface_.contextCreateFromBinary,
                "Invalid function pointer for contextCreateFromBinary.");

  QnnContext_Config_t qnn_context_config = QNN_CONTEXT_CONFIG_INIT;
  ORT_RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, qnn_context_config));
  const QnnContext_Config_t* context_configs[] = {&qnn_context_config, nullptr};

  rt = qnn_interface_.contextCreateFromBinary(backend_handle_,
                                              device_handle_,
                                              context_configs,
                                              static_cast<void*>(buffer),
                                              buffer_length,
                                              &context_,
                                              profile_backend_handle_);
  ORT_RETURN_IF(QNN_SUCCESS != rt, "Failed to create context from binary.");

  // More work to support multiple partition, how to map the graph name in compile to qnn graph name
  // Need the lower level framework to understand EPContext op and pass in the partition_name in fused_node during Compile
  if (1 == graph_count) {
    auto qnn_model_pose = qnn_models.begin();
    ORT_RETURN_IF_ERROR(qnn_model_pose->second->DeserializeGraphInfoFromBinaryInfo(graphs_info[0]));
  } else {
    for (uint32_t i = 0; i < graph_count; ++i) {
      std::string graph_name(graphs_info[i].graphInfoV1.graphName);
      auto qnn_model_pos = qnn_models.find(graph_name);
      ORT_RETURN_IF(qnn_model_pos == qnn_models.end(), graph_name + " does not match any EPContext node names.");
      ORT_RETURN_IF_ERROR(qnn_model_pos->second->DeserializeGraphInfoFromBinaryInfo(graphs_info[i]));
    }
  }

  qnn_sys_interface_.systemContextFree(sys_ctx_handle);
  sys_ctx_handle = nullptr;

  ORT_RETURN_IF_ERROR(ExtractBackendProfilingInfo());
  context_created_ = true;

  LOGS(*logger_, VERBOSE) << "Load from cached QNN Context completed.";
  return Status::OK();
}

Status QnnBackendManager::SetupBackend(const logging::Logger& logger, bool load_from_cached_context) {
  if (backend_setup_completed_) {
    LOGS(logger, VERBOSE) << "Backend setup already!";
    return Status::OK();
  }

  if (qnn_saver_path_.empty()) {
    ORT_RETURN_IF_ERROR(LoadBackend());
  } else {
    ORT_RETURN_IF_ERROR(LoadQnnSaverBackend());
  }

  LOGS(logger, VERBOSE) << "LoadBackend succeed.";

  if (load_from_cached_context) {
    ORT_RETURN_IF_ERROR(LoadQnnSystemLib());
  }

  sdk_build_version_ = GetBackendBuildId();
  LOGS(logger, VERBOSE) << "Backend build version: "
                        << sdk_build_version_;

  SetLogger(&logger);
  LOGS(logger, VERBOSE) << "SetLogger succeed.";

  ORT_RETURN_IF_ERROR(InitializeBackend());
  LOGS(logger, VERBOSE) << "InitializeBackend succeed.";

  ORT_RETURN_IF_ERROR(CreateDevice());
  LOGS(logger, VERBOSE) << "CreateDevice succeed.";

  ORT_RETURN_IF_ERROR(InitializeProfiling());
  LOGS(logger, VERBOSE) << "InitializeProfiling succeed.";

  if (!load_from_cached_context) {
    ORT_RETURN_IF_ERROR(CreateContext());
    LOGS(logger, VERBOSE) << "CreateContext succeed.";
  }

  LOGS(logger, VERBOSE) << "QNN SetupBackend succeed";

  backend_setup_completed_ = true;

  return Status::OK();
}

Status QnnBackendManager::CreateHtpPowerCfgId(uint32_t device_id, uint32_t core_id, uint32_t& htp_power_config_id) {
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

Status QnnBackendManager::SetHtpPowerConfig(uint32_t htp_power_config_client_id,
                                            HtpPerformanceMode htp_performance_mode) {
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  ORT_RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
                "HTP infra type = ", htp_infra->infraType, ", which is not perf infra type.");
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
      ORT_THROW("Invalid performance profile %d", static_cast<int>(htp_performance_mode));
      break;
  }
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr = ObtainNullTermPtrVector(power_configs);
  status = htp_perf_infra.setPowerConfig(htp_power_config_client_id, perf_power_configs_ptr.data());
  ORT_RETURN_IF(QNN_SUCCESS != status, "setPowerConfig failed for HTP performance mode.");

  return Status::OK();
}

Status QnnBackendManager::SetRpcControlLatency(uint32_t htp_power_config_client_id,
                                               uint32_t rpc_control_latency) {
  if (rpc_control_latency != 0) {
    QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
    auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
    ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

    auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
    ORT_RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
                  "HTP infra type = ", htp_infra->infraType, ", which is not perf infra type.");
    QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

    // Set rpc control latency here, but note that v68 doesn't support rpc polling mode.
    constexpr int kNumRpcPollingPowerConfigs = 2;
    std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> rpc_power_configs(kNumRpcPollingPowerConfigs);
    QnnHtpPerfInfrastructure_PowerConfig_t& rpc_control_latency_cfg = rpc_power_configs[0];
    // v68 doesn't support this.
    QnnHtpPerfInfrastructure_PowerConfig_t& rpc_polling_time = rpc_power_configs[1];
    rpc_control_latency_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc_polling_time.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpc_control_latency_cfg.rpcControlLatencyConfig = rpc_control_latency;
    std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr =
        ObtainNullTermPtrVector(rpc_power_configs);
    status = htp_perf_infra.setPowerConfig(htp_power_config_client_id, perf_power_configs_ptr.data());
    ORT_RETURN_IF(QNN_SUCCESS != status, "setPowerConfig failed for RPC control latency.");
  }

  return Status::OK();
}

void QnnBackendManager::Split(std::vector<std::string>& split_string,
                              const std::string& tokenized_string,
                              const char separator) {
  split_string.clear();
  std::istringstream tokenized_string_stream(tokenized_string);
  while (!tokenized_string_stream.eof()) {
    std::string value;
    getline(tokenized_string_stream, value, separator);
    if (!value.empty()) {
      split_string.push_back(value);
    }
  }
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

void QnnBackendManager::ReleaseResources() {
  if (!backend_setup_completed_) {
    return;
  }

  auto result = ReleaseContext();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ReleaseContext.");
  }

  result = ReleaseProfilehandle();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ReleaseProfilehandle.");
  }

  result = ReleaseDevice();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ReleaseDevice.");
  }

  result = ShutdownBackend();
  if (Status::OK() != result) {
    ORT_THROW("Failed to ShutdownBackend.");
  }

  result = TerminateQnnLog();
  if (Status::OK() != result) {
    ORT_THROW("Failed to TerminateQnnLog.");
  }

  if (backend_lib_handle_) {
    result = UnloadLib(backend_lib_handle_);
    if (Status::OK() != result) {
      ORT_THROW("Failed to unload backend library.");
    }
  }

  backend_setup_completed_ = false;

  return;
}

Status QnnBackendManager::ExtractBackendProfilingInfo() {
  if (ProfilingLevel::OFF == profiling_level_merge_ || ProfilingLevel::INVALID == profiling_level_merge_) {
    return Status::OK();
  }

  bool tracelogging_provider_ep_enabled = false;
  const Env& env = Env::Default();
  auto& provider = env.GetTelemetryProvider();
  auto level = provider.Level();
  if (provider.IsEnabled()) {
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)) != 0 && level >= 5) {
      tracelogging_provider_ep_enabled = true;
    }
  }

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
                "Need to specify a cvs file via provider option profiling_file_path if ETW not enabled.");

  ORT_RETURN_IF(nullptr == profile_backend_handle_, "Backend profile handle not valid.");

  const QnnProfile_EventId_t* profile_events{nullptr};
  uint32_t num_events{0};
  auto result = qnn_interface_.profileGetEvents(profile_backend_handle_, &profile_events, &num_events);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile events.");

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

    std::ofstream outfile;
    if (!tracelogging_provider_ep_enabled) {
      // Write to CSV in append mode
      std::ifstream infile(profiling_file_path_.c_str());
      bool exists = infile.good();
      infile.close();

      outfile.open(profiling_file_path_, std::ios_base::app);
      ORT_RETURN_IF(!outfile.is_open(), "Failed to open profiling file: ", profiling_file_path_);
      // If file didn't exist before, write the header
      if (!exists) {
        outfile << "Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier\n";
      }
    }

    for (size_t event_idx = 0; event_idx < num_events; event_idx++) {
      ORT_RETURN_IF_ERROR(
          ExtractProfilingEvent(*(profile_events + event_idx), "ROOT", outfile, backendSupportsExtendedEventData,
                                tracelogging_provider_ep_enabled));
      ORT_RETURN_IF_ERROR(
          ExtractProfilingSubEvents(*(profile_events + event_idx), outfile, backendSupportsExtendedEventData,
                                    tracelogging_provider_ep_enabled));
    }

    if (!tracelogging_provider_ep_enabled) {
      outfile.close();
      LOGS(*logger_, VERBOSE) << "Wrote QNN profiling events (" << num_events << ") to qnn-profiling-data.csv";
    } else {
      LOGS(*logger_, VERBOSE) << "Wrote QNN profiling events (" << num_events << ") to ETW";
    }
  }

  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingSubEvents(
    QnnProfile_EventId_t profile_event_id,
    std::ofstream& outfile,
    bool useExtendedEventData,
    bool tracelogging_provider_ep_enabled) {
  const QnnProfile_EventId_t* profile_sub_events{nullptr};
  uint32_t num_sub_events{0};
  auto result = qnn_interface_.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile sub events.");

  if (num_sub_events > 0) {
    LOGS(*logger_, VERBOSE) << "profile_sub_events: " << profile_sub_events << " num_sub_events: " << num_sub_events;

    for (size_t sub_event_idx = 0; sub_event_idx < num_sub_events; sub_event_idx++) {
      ORT_RETURN_IF_ERROR(
          ExtractProfilingEvent(*(profile_sub_events + sub_event_idx), "SUB-EVENT", outfile, useExtendedEventData,
                                tracelogging_provider_ep_enabled));
      ORT_RETURN_IF_ERROR(
          ExtractProfilingSubEvents(*(profile_sub_events + sub_event_idx), outfile, useExtendedEventData,
                                    tracelogging_provider_ep_enabled));
    }

    LOGS(*logger_, VERBOSE) << "Wrote QNN profiling sub events (" << num_sub_events << ")";
  }

  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingEvent(
    QnnProfile_EventId_t profile_event_id,
    const std::string& eventLevel,
    std::ofstream& outfile,
    bool useExtendedEventData,
    bool tracelogging_provider_ep_enabled) {
  if (useExtendedEventData) {
    return ExtractProfilingEventExtended(profile_event_id, eventLevel, outfile, tracelogging_provider_ep_enabled);
  } else {
    return ExtractProfilingEventBasic(profile_event_id, eventLevel, outfile, tracelogging_provider_ep_enabled);
  }
}

Status QnnBackendManager::ExtractProfilingEventBasic(
    QnnProfile_EventId_t profile_event_id,
    const std::string& eventLevel,
    std::ofstream& outfile,
    bool tracelogging_provider_ep_enabled) {
  QnnProfile_EventData_t event_data;
  auto result = qnn_interface_.profileGetEventData(profile_event_id, &event_data);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(result & 0xFFFF);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode)));

  std::string message = GetEventTypeString(event_data.type);
  std::string unit = GetUnitString(event_data.unit);

#ifndef _WIN32
  tracelogging_provider_ep_enabled = false;
#endif

  if (!tracelogging_provider_ep_enabled) {
    outfile << "UNKNOWN"
            << ","
            << message << ","
            << event_data.value << ","
            << unit << ","
            << "BACKEND"
            << ","
            << eventLevel << ","
            << (event_data.identifier ? event_data.identifier : "NULL") << "\n";
  } else {
#ifdef _WIN32
    LogQnnProfileEventAsTraceLogging(
        (uint64_t)0,
        message,
        std::to_string(event_data.value),
        unit,
        "BACKEND",
        eventLevel,
        (event_data.identifier ? event_data.identifier : "NULL"));
#endif
  }

  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingEventExtended(
    QnnProfile_EventId_t profile_event_id,
    const std::string& eventLevel,
    std::ofstream& outfile,
    bool tracelogging_provider_ep_enabled) {
  QnnProfile_ExtendedEventData_t event_data_extended;
  auto resultGetExtendedEventData = qnn_interface_.profileGetExtendedEventData(profile_event_id, &event_data_extended);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(resultGetExtendedEventData & 0xFFFF);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != errorCode, "Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode)));

  // need to check the version first
  std::string message = GetEventTypeString(event_data_extended.v1.type);
  std::string unit = GetUnitString(event_data_extended.v1.unit);

#ifndef _WIN32
  tracelogging_provider_ep_enabled = false;
#endif

  if (!tracelogging_provider_ep_enabled) {
    // QNN issue, the version number not correct, ticket created
    // if (event_data_extended.version == QNN_PROFILE_DATA_VERSION_1) {
    outfile << event_data_extended.v1.timestamp << ","
            << message << ","
            << ExtractQnnScalarValue(event_data_extended.v1.value) << ","
            << unit << ","
            << "BACKEND"
            << ","
            << eventLevel << ","
            << (event_data_extended.v1.identifier ? event_data_extended.v1.identifier : "NULL") << "\n";
    //}
  } else {
#ifdef _WIN32
    LogQnnProfileEventAsTraceLogging(
        event_data_extended.v1.timestamp,
        message,
        ExtractQnnScalarValue(event_data_extended.v1.value),
        unit,
        "BACKEND",
        eventLevel,
        (event_data_extended.v1.identifier ? event_data_extended.v1.identifier : "NULL"));
#endif
  }

  return Status::OK();
}

#ifdef _WIN32
void QnnBackendManager::LogQnnProfileEventAsTraceLogging(
    uint64_t timestamp,
    const std::string& message,
    const std::string& qnnScalarValue,
    const std::string& unit,
    const std::string& timingSource,
    const std::string& eventLevel,
    const char* eventIdentifier) {
  TraceLoggingWrite(
      telemetry_provider_handle,
      "QNNProfilingEvent",
      TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Profiling)),
      TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
      TraceLoggingValue(timestamp, "Timestamp"),
      TraceLoggingString(message.c_str(), "Message"),
      TraceLoggingString(qnnScalarValue.c_str(), "Value"),
      TraceLoggingString(unit.c_str(), "Unit of Measurement"),
      TraceLoggingString(timingSource.c_str(), "Timing Source"),
      TraceLoggingString(eventLevel.c_str(), "Event Level"),
      TraceLoggingString(eventIdentifier, "Event Identifier"));
}
#endif

const std::string& QnnBackendManager::GetUnitString(QnnProfile_EventUnit_t unitType) {
  const auto& unitStringMap = GetUnitStringMap();
  auto it = unitStringMap.find(unitType);
  if (it != unitStringMap.end()) {
    return it->second;
  }
  static const std::string unknown = "UNKNOWN";
  return unknown;
}

const std::unordered_map<QnnProfile_EventUnit_t, std::string>& QnnBackendManager::GetUnitStringMap() {
  static const std::unordered_map<QnnProfile_EventUnit_t, std::string> unitStringMap = {
      {QNN_PROFILE_EVENTUNIT_MICROSEC, "US"},
      {QNN_PROFILE_EVENTUNIT_BYTES, "BYTES"},
      {QNN_PROFILE_EVENTUNIT_CYCLES, "CYCLES"},
      {QNN_PROFILE_EVENTUNIT_COUNT, "COUNT"},
      {QNN_PROFILE_EVENTUNIT_OBJECT, "OBJECT"},
      {QNN_PROFILE_EVENTUNIT_BACKEND, "BACKEND"}};
  return unitStringMap;
}

const std::string QnnBackendManager::GetEventTypeString(QnnProfile_EventType_t eventType) {
  // Interpret the event type
  switch (eventType) {
    case QNN_PROFILE_EVENTTYPE_INIT:
      return "INIT";
    case QNN_PROFILE_EVENTTYPE_FINALIZE:
      return "FINALIZE";
    case QNN_PROFILE_EVENTTYPE_EXECUTE:
      return "EXECUTE";
    case QNN_PROFILE_EVENTTYPE_NODE:
      return "NODE";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_QUEUE_WAIT:
      return "EXECUTE QUEUE WAIT";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_PREPROCESS:
      return "EXECUTE PREPROCESS";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE:
      return "EXECUTE DEVICE";
    case QNN_PROFILE_EVENTTYPE_EXECUTE_POSTPROCESS:
      return "EXECUTE POSTPROCESS";
    case QNN_PROFILE_EVENTTYPE_DEINIT:
      return "DE-INIT";
    case QNN_PROFILE_EVENTTYPE_BACKEND:
      return "BACKEND";
    default:
      if (eventType > QNN_PROFILE_EVENTTYPE_BACKEND) {
        return "BACKEND";
      }
      return "UNKNOWN";
  }
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

const std::string QnnBackendManager::ExtractQnnScalarValue(const Qnn_Scalar_t& scalar) {
  switch (scalar.dataType) {
    case QNN_DATATYPE_INT_8:
      return std::to_string(static_cast<int>(scalar.int8Value));
    case QNN_DATATYPE_INT_16:
      return std::to_string(scalar.int16Value);
    case QNN_DATATYPE_INT_32:
      return std::to_string(scalar.int32Value);
    case QNN_DATATYPE_INT_64:
      return std::to_string(scalar.int64Value);
    case QNN_DATATYPE_UINT_8:
      return std::to_string(static_cast<unsigned int>(scalar.uint8Value));
    case QNN_DATATYPE_UINT_16:
      return std::to_string(scalar.uint16Value);
    case QNN_DATATYPE_UINT_32:
      return std::to_string(scalar.uint32Value);
    case QNN_DATATYPE_UINT_64:
      return std::to_string(scalar.uint64Value);
    case QNN_DATATYPE_FLOAT_16:
      return std::to_string(scalar.floatValue);
    case QNN_DATATYPE_FLOAT_32:
      return std::to_string(scalar.floatValue);
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_32:
      return std::to_string(scalar.int32Value);  // Assume using int types for signed fixed points.
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return std::to_string(scalar.uint32Value);  // Assume using unsigned int types for unsigned fixed points.
    case QNN_DATATYPE_BOOL_8:
      return scalar.bool8Value ? "true" : "false";
    case QNN_DATATYPE_STRING:
      return scalar.stringValue ? scalar.stringValue : "NULL";
    default:
      return "UNKNOWN";
  }
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
    auto pathstring = Env::Default().GetRuntimePath() + ToPathString(file_name);
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

// TODO: QNN SDK 2.17 crashes for some models/tests on Windows x64 when unloading library.
// Example: ReductionOpTest.ArgMax
#if !defined(_M_AMD64)
  if (FreeLibrary(mod) == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to free library.");
  }
#endif  // !defined(_M_AMD64)
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

}  // namespace qnn
}  // namespace onnxruntime
