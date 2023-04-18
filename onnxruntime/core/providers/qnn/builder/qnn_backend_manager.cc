// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_backend_manager.h"
#include <iostream>
#include "QnnOpDef.h"
#include "DSP/QnnDspPerfInfrastructure.h"
#include "DSP/QnnDspBackend.h"
#include "DSP/QnnDspDevice.h"
#include "DSP/QnnDspCommon.h"
#include "HTP/QnnHtpCommon.h"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

namespace onnxruntime {
namespace qnn {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                          uint32_t* numProviders);

Status QnnBackendManager::LoadBackend() {
  std::string error_msg = "";
  backend_lib_handle_ = LoadLib(backend_path_.c_str(),
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_LOCAL),
                                error_msg);
  ORT_RETURN_IF(nullptr == backend_lib_handle_, "Unable to load backend, error: ", error_msg, " ", DlError());

  // Get QNN Interface
  QnnInterfaceGetProvidersFn_t GetInterfaceProviders{nullptr};
  GetInterfaceProviders = ResolveSymbol<QnnInterfaceGetProvidersFn_t>(backend_lib_handle_, "QnnInterface_getProviders", *logger_);
  ORT_RETURN_IF(nullptr == GetInterfaceProviders, "Failed to get QNN providers!");
  QnnInterface_t** interface_providers{nullptr};
  uint32_t num_providers{0};
  auto result = GetInterfaceProviders((const QnnInterface_t***)&interface_providers, &num_providers);
  ORT_RETURN_IF((QNN_SUCCESS != result || nullptr == interface_providers || 0 == num_providers), "Failed to get interface providers.");

  bool found_valid_interface{false};
  LOGS_DEFAULT(VERBOSE) << "QNN_API_VERSION_MAJOR: " << QNN_API_VERSION_MAJOR
                        << " QNN_API_VERSION_MINOR: " << QNN_API_VERSION_MINOR;
  for (size_t pIdx = 0; pIdx < num_providers; pIdx++) {
    LOGS_DEFAULT(VERBOSE) << "interface_providers major: " << interface_providers[pIdx]->apiVersion.coreApiVersion.major
                          << " interface_providers minor: " << interface_providers[pIdx]->apiVersion.coreApiVersion.minor;
    if (QNN_API_VERSION_MAJOR == interface_providers[pIdx]->apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <= interface_providers[pIdx]->apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_interface_ = interface_providers[pIdx]->QNN_INTERFACE_VER_NAME;
      auto backend_id = interface_providers[pIdx]->backendId;
      if (QNN_BACKEND_ID_DSP == backend_id || QNN_BACKEND_ID_HTP == backend_id) {
        is_npu_backend_ = true;
      }
      LOGS_DEFAULT(INFO) << "Found valid interface, version: " << QNN_API_VERSION_MAJOR
                         << "." << QNN_API_VERSION_MINOR
                         << " backend provider name: " << interface_providers[pIdx]->providerName
                         << " backend id: " << backend_id;
      break;
    }
  }

  ORT_RETURN_IF_NOT(found_valid_interface, "Unable to find a valid interface.");

  return Status::OK();
}

Status QnnBackendManager::InitializeBackend() {
  if (true == backend_initialized_) {
    LOGS_DEFAULT(INFO) << "Backend intialized already.";
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
    LOGS_DEFAULT(INFO) << "Device intialized already.";
    return Status::OK();
  }

  // Create device if its property supported
  if (!IsDevicePropertySupported()) {
    LOGS_DEFAULT(INFO) << "Skip to create device.";
    return Status::OK();
  }

  LOGS_DEFAULT(INFO) << "Create device.";
  if (nullptr != qnn_interface_.deviceCreate) {
    auto result = qnn_interface_.deviceCreate(log_handle_, nullptr, &device_handle_);
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
  if (ProfilingLevel::OFF == profiling_level_ || ProfilingLevel::INVALID == profiling_level_) {
    LOGS_DEFAULT(INFO) << "Profiling turned off.";
    return Status::OK();
  }

  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  if (ProfilingLevel::BASIC == profiling_level_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  } else if (ProfilingLevel::DETAILED == profiling_level_) {
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

Status QnnBackendManager::CreateContext() {
  if (true == context_created_) {
    LOGS_DEFAULT(INFO) << "Context created already.";
    return Status::OK();
  }

  auto result = qnn_interface_.contextCreate(backend_handle_,
                                             device_handle_,
                                             (const QnnContext_Config_t**)&context_config_,
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

Status QnnBackendManager::SetupBackend(const logging::Logger& logger) {
  if (backend_setup_completed_) {
    LOGS(logger, VERBOSE) << "Backend setup already!";
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(LoadBackend());
  LOGS(logger, VERBOSE) << "LoadBackend succeed.";

  LOGS(logger, VERBOSE) << "Backend build version: "
                        << GetBackendBuildId();

  SetLogger(&logger);
  LOGS(logger, VERBOSE) << "SetLogger succeed.";

  ORT_RETURN_IF_ERROR(InitializeBackend());
  LOGS(logger, VERBOSE) << "InitializeBackend succeed.";

  ORT_RETURN_IF_ERROR(CreateDevice());
  LOGS(logger, VERBOSE) << "CreateDevice succeed.";

  ORT_RETURN_IF_ERROR(InitializeProfiling());
  LOGS(logger, VERBOSE) << "InitializeProfiling succeed.";

  ORT_RETURN_IF_ERROR(CreateContext());
  LOGS(logger, VERBOSE) << "CreateContext succeed.";

  // TODO: failed to createPowerConfigId with Qnn v2, need future investigation
  // Disable it for now since it doen't impact any existing feature
  // Also should enable EP options to control the enablement
  // if (set_power_config && profiling_level_ == qnn::ProfilingLevel::OFF) {
  //  ORT_RETURN_IF_ERROR(SetDspPowerConfig());
  //  LOGS(*logger, VERBOSE) << "SetDspPowerConfig succeed.";
  //}

  LOGS(logger, VERBOSE) << "QNN SetupBackend succeed";

  backend_setup_completed_ = true;

  return Status::OK();
}

Status QnnBackendManager::SetDspPowerConfig() {
  QnnDspPerfInfrastructure_PowerConfig_t dcvs_enable;
  dcvs_enable.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
  dcvs_enable.dcvsEnableConfig = 0;  // FALSE
  QnnDspPerfInfrastructure_PowerConfig_t sleep_disable;
  sleep_disable.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_DISABLE;
  sleep_disable.sleepDisableConfig = 1;
  QnnDspPerfInfrastructure_PowerConfig_t dcvs_power_mode;
  dcvs_power_mode.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
  dcvs_power_mode.dcvsPowerModeConfig = QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  QnnDspPerfInfrastructure_PowerConfig_t bus_VCorner_min;
  bus_VCorner_min.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  bus_VCorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  QnnDspPerfInfrastructure_PowerConfig_t bus_VCorner_target;
  bus_VCorner_target.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  bus_VCorner_target.busVoltageCornerTargetConfig = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  QnnDspPerfInfrastructure_PowerConfig_t bus_VCorner_max;
  bus_VCorner_max.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  bus_VCorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  QnnDspPerfInfrastructure_PowerConfig_t core_VCorner_min;
  core_VCorner_min.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  core_VCorner_min.coreVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  QnnDspPerfInfrastructure_PowerConfig_t core_VCorner_target;
  core_VCorner_target.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  core_VCorner_target.coreVoltageCornerTargetConfig = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  QnnDspPerfInfrastructure_PowerConfig_t core_VCorner_max;
  core_VCorner_max.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  core_VCorner_max.coreVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
  QnnDspPerfInfrastructure_PowerConfig_t rpc_control_latency;
  rpc_control_latency.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  rpc_control_latency.rpcControlLatencyConfig = rpc_control_latency_;

  const QnnDspPerfInfrastructure_PowerConfig_t* power_configs[] = {&dcvs_enable, &sleep_disable,
                                                                   &dcvs_power_mode, &bus_VCorner_min,
                                                                   &bus_VCorner_target, &bus_VCorner_max,
                                                                   &core_VCorner_min, &core_VCorner_target,
                                                                   &core_VCorner_max, &rpc_control_latency,
                                                                   nullptr};

  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  QnnDspDevice_Infrastructure_t* dsp_device_infra = static_cast<QnnDspDevice_Infrastructure_t*>(qnn_device_infra);

  uint32_t powerconfig_client_id{0};
  // TODO: failed to createPowerConfigId with Qnn v2, need future investigation
  status = dsp_device_infra->createPowerConfigId(&powerconfig_client_id);
  ORT_RETURN_IF(QNN_SUCCESS != status, "createPowerConfigId failed.");

  status = dsp_device_infra->setPowerConfig(powerconfig_client_id, power_configs);
  ORT_RETURN_IF(QNN_SUCCESS != status, "setPowerConfig failed.");

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
  if (ProfilingLevel::OFF == profiling_level_ || ProfilingLevel::INVALID == profiling_level_) {
    return Status::OK();
  }
  ORT_RETURN_IF(nullptr == profile_backend_handle_, "Backend profile handle not valid.");

  const QnnProfile_EventId_t* profile_events{nullptr};
  uint32_t num_events{0};
  auto result = qnn_interface_.profileGetEvents(profile_backend_handle_, &profile_events, &num_events);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile events.");

  if (num_events > 0) {
    LOGS(*logger_, VERBOSE) << "profile_events: " << profile_events << " num_events: " << num_events;
  }

  for (size_t event_idx = 0; event_idx < num_events; event_idx++) {
    ORT_RETURN_IF_ERROR(ExtractProfilingEvent(*(profile_events + event_idx)));
    ORT_RETURN_IF_ERROR(ExtractProfilingSubEvents(*(profile_events + event_idx)));
  }
  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id) {
  const QnnProfile_EventId_t* profile_sub_events{nullptr};
  uint32_t num_sub_events{0};
  auto result = qnn_interface_.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get profile sub events.");

  if (num_sub_events > 0) {
    LOGS(*logger_, VERBOSE) << "profile_sub_events: " << profile_sub_events << " num_sub_events: " << num_sub_events;
  }

  for (size_t sub_event_idx = 0; sub_event_idx < num_sub_events; sub_event_idx++) {
    ORT_RETURN_IF_ERROR(ExtractProfilingEvent(*(profile_sub_events + sub_event_idx)));
    ORT_RETURN_IF_ERROR(ExtractProfilingSubEvents(*(profile_sub_events + sub_event_idx)));
  }
  return Status::OK();
}

Status QnnBackendManager::ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id) {
  QnnProfile_EventData_t event_data;
  auto result = qnn_interface_.profileGetEventData(profile_event_id, &event_data);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to get provile event data.");

  LOGS(*logger_, VERBOSE) << "Profiling Event Info - Event Type: " << event_data.type
                          << ", Event Value: " << event_data.value
                          << ", Event Identifier: " << event_data.identifier
                          << ", Event Unit: " << event_data.unit;

  return Status::OK();
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

  // search from system lib path first
  HMODULE mod = LoadLibraryExA(file_name, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
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
#endif

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
