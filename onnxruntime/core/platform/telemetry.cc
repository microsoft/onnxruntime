// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry.h"
#include "core/platform/env.h"

namespace onnxruntime {

void LogRuntimeError(uint32_t sessionId, const common::Status& status, const char* file,
                     const char* function, uint32_t line) {
  const Env& env = Env::Default();
  env.GetTelemetryProvider().LogRuntimeError(sessionId, status, file, function, line);
}

void Telemetry::EnableTelemetryEvents() const {
}

void Telemetry::DisableTelemetryEvents() const {
}

void Telemetry::SetLanguageProjection(uint32_t projection) const {
  ORT_UNUSED_PARAMETER(projection);
}

void Telemetry::LogProcessInfo() const {
}

void Telemetry::LogSessionCreationStart() const {
}

void Telemetry::LogEvaluationStop() const {
}

void Telemetry::LogEvaluationStart() const {
}

void Telemetry::LogSessionCreation(uint32_t session_id, int64_t ir_version, const std::string& model_producer_name,
                                   const std::string& model_producer_version, const std::string& model_domain,
                                   const std::unordered_map<std::string, int>& domain_to_version_map,
                                   const std::string& model_graph_name,
                                   const std::unordered_map<std::string, std::string>& model_metadata,
                                   const std::string& loadedFrom, const std::vector<std::string>& execution_provider_ids,
                                   bool use_fp16) const {
  ORT_UNUSED_PARAMETER(session_id);
  ORT_UNUSED_PARAMETER(ir_version);
  ORT_UNUSED_PARAMETER(model_producer_name);
  ORT_UNUSED_PARAMETER(model_producer_version);
  ORT_UNUSED_PARAMETER(model_domain);
  ORT_UNUSED_PARAMETER(domain_to_version_map);
  ORT_UNUSED_PARAMETER(model_graph_name);
  ORT_UNUSED_PARAMETER(model_metadata);
  ORT_UNUSED_PARAMETER(loadedFrom);
  ORT_UNUSED_PARAMETER(execution_provider_ids);
  ORT_UNUSED_PARAMETER(use_fp16);
}

void Telemetry::LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file,
                                const char* function, uint32_t line) const {
  ORT_UNUSED_PARAMETER(session_id);
  ORT_UNUSED_PARAMETER(status);
  ORT_UNUSED_PARAMETER(file);
  ORT_UNUSED_PARAMETER(function);
  ORT_UNUSED_PARAMETER(line);
}

void Telemetry::LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last, int64_t total_run_duration_since_last) const {
  ORT_UNUSED_PARAMETER(session_id);
  ORT_UNUSED_PARAMETER(total_runs_since_last);
  ORT_UNUSED_PARAMETER(total_run_duration_since_last);
}

void Telemetry::LogExecutionProviderEvent(LUID* adapterLuid) const {
  ORT_UNUSED_PARAMETER(adapterLuid);
}

}  // namespace onnxruntime
