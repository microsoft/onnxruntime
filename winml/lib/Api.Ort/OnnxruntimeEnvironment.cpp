// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "OnnxruntimeEnvironment.h"
#include "OnnxruntimeErrors.h"
#include "core/platform/windows/TraceLoggingConfig.h"
#include <evntrace.h>

using namespace Windows::AI ::MachineLearning;

static bool debug_output_ = false;

static void __stdcall WinmlOrtLoggingCallback(void* param, OrtLoggingLevel severity, const char* category,
                                    const char* logger_id, const char* code_location, const char* message) noexcept {
  UNREFERENCED_PARAMETER(param);
  UNREFERENCED_PARAMETER(logger_id);
  // ORT Fatal and Error Messages are logged as Telemetry, rest are non-telemetry.
  switch (severity) {
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL:  //Telemetry
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_CRITICAL),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location),
          TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR:  //Telemetry
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location),
          TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING:
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_WARNING),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO:
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_INFO),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE:
      __fallthrough;  //Default is Verbose too.
    default:
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location));
  }

  if (debug_output_) {
    OutputDebugStringA((std::string(message) + "\r\n").c_str());
  }
}

static void __stdcall WinmlOrtProfileEventCallback(const OrtProfilerEventRecord* profiler_record) noexcept {
  if (profiler_record->category_ == OrtProfilerEventCategory::NODE_EVENT) {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "OnnxRuntimeProfiling",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
        TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
        TraceLoggingString(profiler_record->category_name_, "Category"),
        TraceLoggingInt64(profiler_record->duration_, "Duration (us)"),
        TraceLoggingInt64(profiler_record->time_span_, "Time Stamp (us)"),
        TraceLoggingString(profiler_record->event_name_, "Event Name"),
        TraceLoggingInt32(profiler_record->process_id_, "Process ID"),
        TraceLoggingInt32(profiler_record->thread_id_, "Thread ID"),
        TraceLoggingString(profiler_record->op_name_, "Operator Name"),
        TraceLoggingString(profiler_record->execution_provider_, "Execution Provider"));
  } else {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "OnnxRuntimeProfiling",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
        TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
        TraceLoggingString(profiler_record->category_name_, "Category"),
        TraceLoggingInt64(profiler_record->duration_, "Duration (us)"),
        TraceLoggingInt64(profiler_record->time_span_, "Time Stamp (us)"),
        TraceLoggingString(profiler_record->event_name_, "Event Name"),
        TraceLoggingInt32(profiler_record->process_id_, "Process ID"),
        TraceLoggingInt32(profiler_record->thread_id_, "Thread ID"));
  }
}

OnnxruntimeEnvironment::OnnxruntimeEnvironment(const OrtApi* ort_api) : ort_env_(nullptr, nullptr) {
  OrtEnv* ort_env = nullptr;
  THROW_IF_NOT_OK_MSG(ort_api->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env),
                      ort_api);
  ort_env_ = UniqueOrtEnv(ort_env, ort_api->ReleaseEnv);

  // Configure the environment with the winml logger
  auto winml_adapter_api = OrtGetWinMLAdapter(ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->EnvConfigureCustomLoggerAndProfiler(ort_env_.get(),
                                                                             &WinmlOrtLoggingCallback, &WinmlOrtProfileEventCallback, nullptr,
                                                                             OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env),
                      ort_api);

  THROW_IF_NOT_OK_MSG(winml_adapter_api->OverrideSchema(), ort_api);
}

HRESULT OnnxruntimeEnvironment::GetOrtEnvironment(_Out_ OrtEnv** ort_env) {
  *ort_env = ort_env_.get();
  return S_OK;
}

HRESULT OnnxruntimeEnvironment::EnableDebugOutput(bool is_enabled) {
  debug_output_ = is_enabled;
  return S_OK;
}