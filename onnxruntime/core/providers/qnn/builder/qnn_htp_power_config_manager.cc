// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/builder/qnn_htp_power_config_manager.h"

#include <vector>

#include <QnnInterface.h>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace power {

HtpPowerConfigManager::HtpPowerConfigManager(const Ort::Logger& logger) : logger_(logger) {
  constexpr int kMaxNumConfigs = 3;
  power_configs_.reserve(kMaxNumConfigs);
}

HtpPowerConfigManager::~HtpPowerConfigManager() {}

Ort::Status HtpPowerConfigManager::AddRpcPollingTime(uint32_t rpc_polling_time) {
  RETURN_IF(rpc_polling_time > kMaxRpcPolling,
            ("Cannot set RPC polling time to " + std::to_string(rpc_polling_time) +
             ". Max allowable RPC polling time is: " + std::to_string(kMaxRpcPolling))
                .c_str());

  RETURN_IF(rpc_polling_time_set_, "There is already a pending RPC polling time config");

  if (rpc_polling_time == last_set_rpc_polling_time_) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Requested rpc polling time is the same as last set (" + std::to_string(last_set_rpc_polling_time_) + "). Ignoring request").c_str());
  } else {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Updating rpc polling time to: " + std::to_string(rpc_polling_time) + "us.").c_str());
    auto& rpc_polling_time_cfg = power_configs_.emplace_back();
    rpc_polling_time_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpc_polling_time_cfg.rpcPollingTimeConfig = rpc_polling_time;

    last_set_rpc_polling_time_ = rpc_polling_time;
    rpc_polling_time_set_ = true;
  }
  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::AddRpcControlLatency(uint32_t rpc_control_latency) {
  RETURN_IF(rpc_control_latency_set_, "There is already a pending RPC control latency config");
  if (rpc_control_latency == last_set_rpc_control_latency_) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Requested rpc control latency is the same as last set (" +
                 std::to_string(last_set_rpc_control_latency_) + "). Ignoring request")
                    .c_str());
  } else {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Updating rpc control latency to: " + std::to_string(rpc_control_latency) + "us.").c_str());
    auto& rpc_control_latency_cfg = power_configs_.emplace_back();
    rpc_control_latency_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc_control_latency_cfg.rpcControlLatencyConfig = rpc_control_latency;

    last_set_rpc_control_latency_ = rpc_control_latency;
    rpc_control_latency_set_ = true;
  }

  return Ort::Status();
}

static std::string_view PerformanceModeToString(HtpPerformanceMode htp_performance_mode) {
  constexpr std::array<std::pair<HtpPerformanceMode, std::string_view>, 10> perf_string_map = {{{HtpPerformanceMode::kHtpDefault, "default"},
                                                                                                {HtpPerformanceMode::kHtpSustainedHighPerformance, "sustained_high_performance"},
                                                                                                {HtpPerformanceMode::kHtpBurst, "burst"},
                                                                                                {HtpPerformanceMode::kHtpHighPerformance, "high_performance"},
                                                                                                {HtpPerformanceMode::kHtpPowerSaver, "power_saver"},
                                                                                                {HtpPerformanceMode::kHtpLowPowerSaver, "low_power_saver"},
                                                                                                {HtpPerformanceMode::kHtpHighPowerSaver, "high_power_saver"},
                                                                                                {HtpPerformanceMode::kHtpLowBalanced, "low_balanced"},
                                                                                                {HtpPerformanceMode::kHtpBalanced, "balanced"},
                                                                                                {HtpPerformanceMode::kHtpExtremePowerSaver, "extreme_power_saver"}}};

  auto it = std::find_if(perf_string_map.begin(), perf_string_map.end(),
                         [htp_performance_mode](const auto& mapping) {
                           return mapping.first == htp_performance_mode;
                         });

  if (it != perf_string_map.end()) {
    return it->second;
  }

  return "UNKNOWN";
}

Ort::Status HtpPowerConfigManager::AddHtpPerformanceMode(HtpPerformanceMode htp_performance_mode,
                                                         uint32_t htp_power_config_client_id) {
  RETURN_IF(htp_performance_mode_set_, "There is already a pending HTP performance mode config");
  if (htp_performance_mode == last_set_htp_performance_mode_) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Requested htp performance mode is the same as last set (" +
                 std::string(PerformanceModeToString(last_set_htp_performance_mode_)) +
                 "). Ignoring request")
                    .c_str());
  } else {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Updating htp performance mode to: " +
                 std::string(PerformanceModeToString(htp_performance_mode)) + ".")
                    .c_str());

    QnnHtpPerfInfrastructure_PowerConfig_t htp_performance_cfg{};
    RETURN_IF_ERROR(SetHtpPerformancePowerConfig(htp_performance_cfg,
                                                 htp_power_config_client_id,
                                                 htp_performance_mode));

    power_configs_.emplace_back(std::move(htp_performance_cfg));

    last_set_htp_performance_mode_ = htp_performance_mode;
    htp_performance_mode_set_ = true;
  }

  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::SetPowerConfig(uint32_t htp_power_config_client_id,
                                                  const QNN_INTERFACE_VER_TYPE& qnn_interface) {
  if (!power_configs_.empty()) {
    QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
    auto status = qnn_interface.deviceGetInfrastructure(&qnn_device_infra);
    RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

    auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
    RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
              ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
    QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

    std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr;

    for (const auto& power_config : power_configs_) {
      perf_power_configs_ptr.push_back(&power_config);
    }
    perf_power_configs_ptr.push_back(nullptr);

    status = htp_perf_infra.setPowerConfig(htp_power_config_client_id, perf_power_configs_ptr.data());
    RETURN_IF(QNN_SUCCESS != status, "SetPowerConfig failed.");

    rpc_polling_time_set_ = false;
    rpc_control_latency_set_ = false;
    htp_performance_mode_set_ = false;
    power_configs_.clear();
  } else {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "SetPowerConfig called but no configs to be set.");
  }

  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::SetHtpPerformancePowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
                                                                uint32_t htp_power_config_client_id,
                                                                const HtpPerformanceMode& htp_performance_mode) {
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = power_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  // choose performance mode
  switch (htp_performance_mode) {
    case HtpPerformanceMode::kHtpBurst:
    case HtpPerformanceMode::kHtpSustainedHighPerformance:
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

  return Ort::Status();
}

}  // namespace power
}  // namespace qnn
}  // namespace onnxruntime
