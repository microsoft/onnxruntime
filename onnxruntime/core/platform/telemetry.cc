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

bool Telemetry::IsEnabled() const {
  return false;
}

// Get the current logging level
// The Level defined as uchar is coming from the ETW Enable callback in TraceLoggingRegisterEx.
unsigned char Telemetry::Level() const {
  return 0;
}

// Get the current keyword
uint64_t Telemetry::Keyword() const {
  return 0;
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
                                   const std::string& model_file_name,
                                   const std::string& model_graph_name,
                                   const std::string& model_weight_type,
                                   const std::string& model_graph_hash,
                                   const std::string& model_weight_hash,
                                   const std::unordered_map<std::string, std::string>& model_metadata,
                                   const std::string& loadedFrom, const std::vector<std::string>& execution_provider_ids,
                                   bool use_fp16, bool captureState) const {
  ORT_UNUSED_PARAMETER(session_id);
  ORT_UNUSED_PARAMETER(ir_version);
  ORT_UNUSED_PARAMETER(model_producer_name);
  ORT_UNUSED_PARAMETER(model_producer_version);
  ORT_UNUSED_PARAMETER(model_domain);
  ORT_UNUSED_PARAMETER(domain_to_version_map);
  ORT_UNUSED_PARAMETER(model_file_name);
  ORT_UNUSED_PARAMETER(model_graph_name);
  ORT_UNUSED_PARAMETER(model_weight_type);
  ORT_UNUSED_PARAMETER(model_graph_hash);
  ORT_UNUSED_PARAMETER(model_weight_hash);
  ORT_UNUSED_PARAMETER(model_metadata);
  ORT_UNUSED_PARAMETER(loadedFrom);
  ORT_UNUSED_PARAMETER(execution_provider_ids);
  ORT_UNUSED_PARAMETER(use_fp16);
  ORT_UNUSED_PARAMETER(captureState);
}

void Telemetry::LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file,
                                const char* function, uint32_t line) const {
  ORT_UNUSED_PARAMETER(session_id);
  ORT_UNUSED_PARAMETER(status);
  ORT_UNUSED_PARAMETER(file);
  ORT_UNUSED_PARAMETER(function);
  ORT_UNUSED_PARAMETER(line);
}

void Telemetry::LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last, int64_t total_run_duration_since_last,
                               std::unordered_map<int64_t, long long> duration_per_batch_size) const {
  ORT_UNUSED_PARAMETER(session_id);
  ORT_UNUSED_PARAMETER(total_runs_since_last);
  ORT_UNUSED_PARAMETER(total_run_duration_since_last);
  ORT_UNUSED_PARAMETER(duration_per_batch_size);
}

void Telemetry::LogExecutionProviderEvent(LUID* adapterLuid) const {
  ORT_UNUSED_PARAMETER(adapterLuid);
}

void Telemetry::LogDriverInfoEvent(const std::string_view device_class,
                                   const std::wstring_view& driver_names,
                                   const std::wstring_view& driver_versions) const {
  ORT_UNUSED_PARAMETER(device_class);
  ORT_UNUSED_PARAMETER(driver_names);
  ORT_UNUSED_PARAMETER(driver_versions);
}

void Telemetry::LogAutoEpSelection(uint32_t session_id, const std::string& selection_policy,
                                   const std::vector<std::string>& requested_execution_provider_ids,
                                   const std::vector<std::string>& available_execution_provider_ids) const {
  ORT_UNUSED_PARAMETER(session_id);
  ORT_UNUSED_PARAMETER(selection_policy);
  ORT_UNUSED_PARAMETER(requested_execution_provider_ids);
  ORT_UNUSED_PARAMETER(available_execution_provider_ids);
}

void Telemetry::LogProviderOptions(const std::string& provider_id,
                                   const std::string& provider_options_string,
                                   bool captureState) const {
  ORT_UNUSED_PARAMETER(provider_id);
  ORT_UNUSED_PARAMETER(provider_options_string);
  ORT_UNUSED_PARAMETER(captureState);
}

}  // namespace onnxruntime
