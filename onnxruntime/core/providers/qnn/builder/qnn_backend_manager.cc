// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_backend_manager.h"
#include <iostream>
#include "QnnOpDef.h"
#include "DSP/QnnDspPerfInfrastructure.h"
#include "DSP/QnnDspBackend.h"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

namespace onnxruntime {
namespace qnn {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t** providerList,
                                                          uint32_t* numProviders);

template <class T>
static inline T resolveSymbol(LIBTYPE libHandle, const char* sym, const logging::Logger& logger) {
  T ptr = (T)LIBFUNC(libHandle, sym);
  if (ptr == nullptr) {
    LOGS(logger, ERROR) << "Unable to access symbol: " << sym << ". dlerror(): " << DLERROR();
  }
  return ptr;
}

Status QnnBackendManager::LoadBackend() {
  backend_handle_ = OPENLIB(backend_path_.c_str());
  ORT_RETURN_IF(nullptr == backend_handle_, "Unable to load backend. dlerror():", DLERROR());

  // Get QNN Interface
  QnnInterfaceGetProvidersFn_t GetInterfaceProviders{nullptr};
  GetInterfaceProviders = resolveSymbol<QnnInterfaceGetProvidersFn_t>(backend_handle_, "QnnInterface_getProviders", *logger_);
  ORT_RETURN_IF(nullptr == GetInterfaceProviders, "Failed to get QNN providers!");
  QnnInterface_t* interface_providers{nullptr};
  uint32_t num_providers{0};
  auto result = GetInterfaceProviders((const QnnInterface_t**)&interface_providers, &num_providers);
  ORT_RETURN_IF((QNN_SUCCESS != result || nullptr == interface_providers || 0 == num_providers), "Failed to get interface providers.");

  bool found_valid_interface{false};
  LOGS_DEFAULT(VERBOSE) << "QNN_API_VERSION_MAJOR: " << QNN_API_VERSION_MAJOR
                        << " QNN_API_VERSION_MINOR: " << QNN_API_VERSION_MINOR;
  for (size_t pIdx = 0; pIdx < num_providers; pIdx++) {
    LOGS_DEFAULT(VERBOSE) << "interface_providers major: " << interface_providers[pIdx].apiVersion.coreApiVersion.major
                          << " interface_providers minor: " << interface_providers[pIdx].apiVersion.coreApiVersion.minor;
    if (QNN_API_VERSION_MAJOR == interface_providers[pIdx].apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <= interface_providers[pIdx].apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_interface_ = interface_providers[pIdx].QNN_INTERFACE_VER_NAME;
      LOGS_DEFAULT(INFO) << "Found valid interface, version: " << QNN_API_VERSION_MAJOR
                         << "." << QNN_API_VERSION_MINOR;
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

  auto result = qnn_interface_.backendInitialize((const QnnBackend_Config_t**)backend_config_);
  ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != result, "Failed to initialize backend");

  backend_initialized_ = true;
  return Status::OK();
}

Status QnnBackendManager::ShutdownBackend() {
  LOGS_DEFAULT(VERBOSE) << "Terminate Qnn backend.";
  if (false == backend_initialized_) {
    LOGS_DEFAULT(INFO) << "Backend not intialized, no need to terminate it.";
    return Status::OK();
  }

  ORT_RETURN_IF(QNN_BACKEND_NO_ERROR != qnn_interface_.backendTerminate(),
                "Failed to shutdown backend!");

  LOGS_DEFAULT(VERBOSE) << "Terminate Qnn backend succeed.";
  backend_initialized_ = false;

  return Status::OK();
}

Status QnnBackendManager::InitializeProfiling() {
  if (ProfilingLevel::OFF == profiling_level_ || ProfilingLevel::INVALID == profiling_level_) {
    LOGS_DEFAULT(INFO) << "Profiling turned off.";
    return Status::OK();
  }

  LOGS_DEFAULT(INFO) << "Profiling turned on; level = " << profiling_level_;
  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  if (ProfilingLevel::BASIC == profiling_level_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  } else if (ProfilingLevel::DETAILED == profiling_level_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
  }
  auto result = qnn_interface_.profileCreate(qnn_profile_level, &profile_backend_handle_);
  ORT_RETURN_IF(QNN_PROFILE_NO_ERROR != result, "Failed to create QNN profile!");

  return Status::OK();
}

Status QnnBackendManager::ReleaseProfilehandle() {
  // Free Profiling object if it was created
  if (nullptr != profile_backend_handle_) {
    LOGS_DEFAULT(VERBOSE) << "Release backend profile handle.";
    if (QNN_PROFILE_NO_ERROR != qnn_interface_.profileFree(profile_backend_handle_)) {
      LOGS_DEFAULT(ERROR) << "Could not free backend profile handle.";
    }
  }
  profile_backend_handle_ = nullptr;
  LOGS_DEFAULT(VERBOSE) << "Release backend profile handle succeed.";

  return Status::OK();
}

Status QnnBackendManager::CreateContext() {
  if (true == context_created_) {
    LOGS_DEFAULT(INFO) << "Context created already.";
    return Status::OK();
  }

  auto result = qnn_interface_.contextCreate((const QnnContext_Config_t**)&context_config_, &context_);

  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to create context.");

  context_created_ = true;
  return Status::OK();
}

Status QnnBackendManager::ReleaseContext() {
  LOGS_DEFAULT(VERBOSE) << "Release context.";
  if (false == context_created_) {
    LOGS_DEFAULT(INFO) << "Context not created, no need to be freed.";
    return Status::OK();
  }

  auto result = qnn_interface_.contextFree(context_, profile_backend_handle_);
  ORT_RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to release context.");

  LOGS_DEFAULT(VERBOSE) << "Release context succeed.";
  context_created_ = false;
  return Status::OK();
}

Status QnnBackendManager::SetupBackend(const logging::Logger* logger) {
  if (backend_setup_completed_) {
    LOGS(*logger, VERBOSE) << "Backend setup already!";
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(LoadBackend());
  LOGS(*logger, VERBOSE) << "LoadBackend succeed.";

  LOGS(*logger, VERBOSE) << "Backend build version: "
                         << GetBackendBuildId();

  SetLogger(logger);
  LOGS(*logger, VERBOSE) << "SetLogger succeed.";

  ORT_RETURN_IF_ERROR(InitializeBackend());
  LOGS(*logger, VERBOSE) << "InitializeBackend succeed.";

  ORT_RETURN_IF_ERROR(InitializeProfiling());
  LOGS(*logger, VERBOSE) << "InitializeProfiling succeed.";

  ORT_RETURN_IF_ERROR(CreateContext());
  LOGS(*logger, VERBOSE) << "CreateContext succeed.";

  if (is_dsp_backend_ && profiling_level_ == qnn::ProfilingLevel::OFF) {
    ORT_RETURN_IF_ERROR(SetDspPowerConfig());
    LOGS(*logger, VERBOSE) << "SetDspPowerConfig succeed.";
  }
  LOGS(*logger, VERBOSE) << "QNN SetupBackend succeed";

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

  QnnBackend_PerfInfrastructure_t qnn_backend_perf_infra = nullptr;
  auto status = qnn_interface_.backendGetPerfInfrastructure(&qnn_backend_perf_infra);
  ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  QnnDspBackend_PerfInfrastructure_t* dsp_perf_infra = static_cast<QnnDspBackend_PerfInfrastructure_t*>(qnn_backend_perf_infra);

  uint32_t powerconfig_client_id{0};
  status = dsp_perf_infra->createPowerConfigId(&powerconfig_client_id);
  ORT_RETURN_IF(QNN_SUCCESS != status, "createPowerConfigId failed.");

  status = dsp_perf_infra->setPowerConfig(powerconfig_client_id, power_configs);
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
    LOGS(*logger_, ERROR) << "Failed to ReleaseContext.";
  }

  result = ReleaseProfilehandle();
  if (Status::OK() != result) {
    LOGS(*logger_, ERROR) << "Failed to ReleaseProfilehandle.";
  }

  result = ShutdownBackend();
  if (Status::OK() != result) {
    LOGS(*logger_, ERROR) << "Failed to ShutdownBackend.";
  }

  TerminateQnnLog();

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

}  // namespace qnn
}  // namespace onnxruntime
