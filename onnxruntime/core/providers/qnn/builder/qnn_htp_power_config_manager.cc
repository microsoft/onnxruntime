#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

#include <vector>

#include <QnnInterface.h>

namespace onnxruntime {
namespace qnn {
namespace powerconfig {

HtpPowerConfigManager::HtpPowerConfigManager() {
  constexpr int kMaxNumConfigs = 3;
  power_configs_.reserve(kMaxNumConfigs);
};
HtpPowerConfigManager::~HtpPowerConfigManager() {}

Status HtpPowerConfigManager::AddRpcPollingTime(const uint32_t& rpc_polling_time) {
  ORT_RETURN_IF(rpc_polling_time > kMaxRpcPolling, "Cannot set RPC polling time to ",
                std::to_string(rpc_polling_time),
                ". Max allowable RPC pollng time is: ",
                std::to_string(kMaxRpcPolling));

  ORT_RETURN_IF(rpc_polling_time_set_, "There is already a pending RPC polling time config");

  if (rpc_polling_time == last_set_rpc_polling_time_) {
    LOGS_DEFAULT(VERBOSE) << "Requested rpc polling time is the same as last set ("
                          << last_rpc_polling_time_
                          << "). Ignoring request";
  } else {
    auto& rpc_polling_time_cfg = power_configs_.emplace_back();
    rpc_polling_time_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpc_polling_time_cfg.rpcPollingTimeConfig = rpc_polling_time;

    last_set_rpc_polling_time_ = rpc_polling_time;
    rpc_polling_time_set_ = true;
  }
  return Status::OK();
}

Status HtpPowerConfigManager::AddRpcControlLatency(const uint32_t& rpc_control_latency) {
  ORT_RETURN_IF(rpc_control_latency_set_, "There is already a pending RPC control latency config");
  if (rpc_control_latency == last_set_rpc_control_latency_) {
    LOGS_DEFAULT(VERBOSE) << "Requested rpc control latency is the same as last set ("
                          << last_rpc_control_latency_
                          << "). Ignoring request";
  } else {
    auto& rpc_control_latency_cfg = power_configs_.emplace_back();
    rpc_control_latency_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc_control_latency_cfg.rpcControlLatencyConfig = rpc_control_latency;

    last_set_rpc_control_latency_ = rpc_control_latency;
    rpc_control_latency_set_ = true;
  }

  return Status::OK();
}

Status HtpPowerConfigManager::AddHtpPerformanceMode(const HtpPerformanceMode& htp_performance_mode) {
  ORT_RETURN_IF(htp_performance_mode_set_, "There is already a pending HTP performance mode config");
  if (htp_performance_mode == last_set_htp_performance_mode_) {
    LOGS_DEFAULT(VERBOSE) << "Requested htp performance mode is the same as last set ("
                          << last_set_htp_performance_mode_
                          << "). Ignoring request";
  } else {
    auto& htp_performance_cfg = power_configs_.emplace_back();
    ORT_RETURN_IF_ERROR(SetHtpPerformancePowerConfig(htp_performance_cfg));

    last_set_htp_performance_mode_ = htp_performance_mode;
    htp_performance_mode_set_ = true;
  }

  return Status::OK();
}

Status HtpPowerConfigManager::SetPowerConfig(uint32_t htp_power_config_client_id,
                                             const QNN_INTERFACE_VER_TYPE& qnn_interface) {
  if (power_configs_.size() < 0) {
    QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
    auto status = qnn_interface.deviceGetInfrastructure(&qnn_device_infra);
    ORT_RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

    auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
    ORT_RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
                  "HTP infra type = ", htp_infra->infraType, ", which is not perf infra type.");
    QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

    std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr = {power_configs_, nullptr};
    status = htp_perf_infra.setPowerConfig(htp_power_config_client_id, perf_power_configs_ptr.data());
    ORT_RETURN_IF(QNN_SUCCESS != status, "SetPowerConfig failed.");

    power_configs_.clear();
  } else {
    LOGS_DEFAULT(WARNING) << "SetPowerConfig called but no configs to be set.";
  }

  return Status::OK();
}

Status HtpPowerConfigManager::SetHtpPerformancePowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
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

  return Status::OK();
}

}  // namespace powerconfig
}  // namespace qnn
}  // namespace onnxruntime