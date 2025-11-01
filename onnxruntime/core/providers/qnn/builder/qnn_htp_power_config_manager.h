#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

#include <vector>

#include <QnnInterface.h>

namespace onnxruntime {
namespace qnn {
namespace power {

// Manages staging of any new power configurations and
// updates power configurations for the HTP backend
class HtpPowerConfigManager {
  HtpPowerConfigManager();
  ~HtpPowerConfigManager();

  // Stages a new rpc polling time for next power config update
  // If the value is the same as the last previously set, then
  // there will be no new rpc polling time staged
  Status AddRpcPollingTime(const uint32_t& rpc_polling_time);

  // Stages a new rpc control latency for next power config update
  // If the value is the same as the last previously set, then
  // there will be no new rpc control latency staged
  Status AddRpcControlLatency(const uint32_t& rpc_control_latency);

  // Stages a new performance mode for next power config update
  // If the value is the same as the last previously set, then
  // there will be no new performance mode staged
  Status AddHtpPerformanceMode(const HtpPerformanceMode& htp_performance_mode);

  // Takes all configs staged for update and attempts to update
  // the HTP power configurations. If there is nothing staged,
  // then no attempt will be made.
  Status SetPowerConfig(uint32_t htp_power_config_client_id,
                        const QNN_INTERFACE_VER_TYPE& qnn_interface);

 private:
  // Sets voltage corner votes for HTP based on the given performance mode
  Status SetHtpPerformancePowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
                                      const HtpPerformanceMode& htp_performance_mode);

  uint32_t last_set_rpc_polling_time_ = kDisableRpcPolling;
  uint32_t last_set_rpc_control_latency_ = kDisableRpcControlLatency;
  HtpPerformanceMode last_set_htp_performance_mode_ = HtpPerformanceMode::kHtpDefault;

  bool rpc_polling_time_set_ = false;
  bool rpc_control_latency_set_ = false;
  bool htp_performance_mode_set_ = false;

  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs_;
};
}  // namespace powerconfig
}  // namespace qnn
}  // namespace onnxruntime