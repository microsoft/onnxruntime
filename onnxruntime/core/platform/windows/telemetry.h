// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/telemetry.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/windows/TraceLoggingConfig.h"
#include <atomic>

namespace onnxruntime {

/**
 * derives and implments a Telemetry provider on Windows
 */
class WindowsTelemetry : public Telemetry {
 public:
  // these are allowed to be created, WindowsEnv will create one
  WindowsTelemetry();
  ~WindowsTelemetry();

  void EnableTelemetryEvents() const override;
  void DisableTelemetryEvents() const override;
  void SetLanguageProjection(uint32_t projection) const override;

  void LogProcessInfo() const override;

  void LogSessionCreationStart() const override;

  void LogEvaluationStop() const override;

  void LogEvaluationStart() const override;

  void LogSessionCreation(uint32_t session_id, int64_t ir_version, const std::string& model_producer_name,
                          const std::string& model_producer_version, const std::string& model_domain,
                          const std::unordered_map<std::string, int>& domain_to_version_map,
                          const std::string& model_graph_name,
                          const std::unordered_map<std::string, std::string>& model_metadata,
                          const std::string& loadedFrom, const std::vector<std::string>& execution_provider_ids,
                          bool use_fp16) const override;

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file,
                       const char* function, uint32_t line) const override;

  void LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last, int64_t total_run_duration_since_last) const override;

  void LogExecutionProviderEvent(LUID* adapterLuid) const override;

 private:
  static OrtMutex mutex_;
  static uint32_t global_register_count_;
  static bool enabled_;
  static uint32_t projection_;
};

}  // namespace onnxruntime
