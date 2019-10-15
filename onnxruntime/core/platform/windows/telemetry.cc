// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/telemetry.h"
#include "core/common/version.h"

// ETW includes
// need space after Windows.h to prevent clang-format re-ordering breaking the build.
// TraceLoggingProvider.h must follow Windows.h
#include <Windows.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26440)  // Warning C26440 from TRACELOGGING_DEFINE_PROVIDER
#endif

#include <TraceLoggingProvider.h>
#include <evntrace.h>

namespace onnxruntime {

namespace {
TRACELOGGING_DEFINE_PROVIDER(telemetry_provider_handle, "Microsoft.ML.ONNXRuntime",
                             // {3a26b1ff-7484-7484-7484-15261f42614d}
                             (0x3a26b1ff, 0x7484, 0x7484, 0x74, 0x84, 0x15, 0x26, 0x1f, 0x42, 0x61, 0x4d),
                             TraceLoggingOptionMicrosoftTelemetry());
}  // namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif

OrtMutex WindowsTelemetry::mutex_;
uint32_t WindowsTelemetry::global_register_count_ = 0;
bool WindowsTelemetry::enabled_ = false;


WindowsTelemetry::WindowsTelemetry() {
  std::lock_guard<OrtMutex> lock(mutex_);
  if (global_register_count_ == 0) {
    // TraceLoggingRegister is fancy in that you can only register once GLOBALLY for the whole process
    HRESULT hr = TraceLoggingRegister(telemetry_provider_handle);
    if (SUCCEEDED(hr)) {
      global_register_count_ += 1;
    }
  }
}

WindowsTelemetry::~WindowsTelemetry() {
  std::lock_guard<OrtMutex> lock(mutex_);
  if (global_register_count_ > 0) {
    global_register_count_ -= 1;
    if (global_register_count_ == 0) {
      TraceLoggingUnregister(telemetry_provider_handle);
    }
  }
}

void WindowsTelemetry::EnableTelemetryEvents() const {
  enabled_ = true;
}

void WindowsTelemetry::DisableTelemetryEvents() const {
  enabled_ = false;
}

void WindowsTelemetry::LogProcessInfo() const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  static std::atomic<bool> process_info_logged;

  // did we already log the process info?  we only need to log it once
  if (process_info_logged.exchange(true))
    return;

  TraceLoggingWrite(telemetry_provider_handle,
                    "ProcessInfo",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingString(ONNXRUNTIME_VERSION_STRING, "runtimeVersion"),
                    TraceLoggingBool(true, "isRedist"));

  process_info_logged = true;
}

void WindowsTelemetry::LogSessionCreation(uint32_t session_id, int64_t ir_version, const std::string& model_producer_name,
                                          const std::string& model_producer_version, const std::string& model_domain,
                                          const std::unordered_map<std::string, int>& domain_to_version_map,
                                          const std::string& model_graph_name,
                                          const std::unordered_map<std::string, std::string>& model_metadata,
                                          const std::string& loadedFrom, const std::vector<std::string>& execution_provider_ids) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  // build the strings we need

  std::string domain_to_verison_string;
  bool first = true;
  for (auto& i : domain_to_version_map) {
    if (first) {
      first = false;
    } else {
      domain_to_verison_string += ',';
    }
    domain_to_verison_string += i.first;
    domain_to_verison_string += '=';
    domain_to_verison_string += std::to_string(i.second);
  }

  std::string model_metadata_string;
  first = true;
  for (auto& i : model_metadata) {
    if (first) {
      first = false;
    } else {
      model_metadata_string += ',';
    }
    model_metadata_string += i.first;
    model_metadata_string += '=';
    model_metadata_string += i.second;
  }

  std::string execution_provider_string;
  first = true;
  for (auto& i : execution_provider_ids) {
    if (first) {
      first = false;
    } else {
      execution_provider_string += ',';
    }
    execution_provider_string += i;
  }

  TraceLoggingWrite(telemetry_provider_handle,
                    "SessionCreation",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingInt64(ir_version, "irVersion"),
                    TraceLoggingString(model_producer_name.c_str(), "modelProducerName"),
                    TraceLoggingString(model_producer_version.c_str(), "modelProducerVersion"),
                    TraceLoggingString(model_domain.c_str(), "modelDomain"),
                    TraceLoggingString(domain_to_verison_string.c_str(), "domainToVersionMap"),
                    TraceLoggingString(model_graph_name.c_str(), "modelGraphName"),
                    TraceLoggingString(model_metadata_string.c_str(), "modelMetaData"),
                    TraceLoggingString(loadedFrom.c_str(), "loadedFrom"),
                    TraceLoggingString(execution_provider_string.c_str(), "executionProviderIds"));
}

void WindowsTelemetry::LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file,
                                       const char* function, uint32_t line) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  TraceLoggingWrite(telemetry_provider_handle,
                    "RuntimeError",
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingUInt32(status.Code(), "errorCode"),
                    TraceLoggingUInt32(status.Category(), "errorCategory"),
                    TraceLoggingString(status.ErrorMessage().c_str(), "errorMessage"),
                    TraceLoggingString(file, "file"),
                    TraceLoggingString(function, "function"),
                    TraceLoggingInt32(line, "line"));
}

void WindowsTelemetry::LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last, int64_t total_run_duration_since_last) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  TraceLoggingWrite(telemetry_provider_handle,
                    "RuntimePerf",
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingUInt32(total_runs_since_last, "totalRuns"),
                    TraceLoggingInt64(total_run_duration_since_last, "totalRunDuration"));
}

}  // namespace onnxruntime
