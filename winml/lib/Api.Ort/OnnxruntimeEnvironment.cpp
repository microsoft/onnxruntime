// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"
#include "OnnxruntimeEnvironment.h"
#include "core/platform/windows/TraceLoggingConfig.h"
#include <evntrace.h>

using namespace Windows::AI ::MachineLearning;

static bool debug_output_ = false;

static void WinmlOrtLoggingCallback(void* param, OrtLoggingLevel severity, const char* category,
                             const char* logger_id, const char* code_location, const char* message) {
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
          TraceLoggingString(""),  // TODO figure out message location: message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str()
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
          TraceLoggingString(""),  // TODO figure out message location: message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str()
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
          TraceLoggingString(""));  // TODO figure out message location: message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str()
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
          TraceLoggingString(""));  // TODO figure out message location: message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str()
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
          TraceLoggingString(""));  // TODO figure out message location: message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str()
  }

  if (debug_output_) {
    OutputDebugStringA((std::string(message) + "\r\n").c_str());
  }
}

static HRESULT OverrideSchemaInferenceFunctions(const OrtApi* ort_api) {
  // This only makes sense for ORT.
  // Before creating any models, we ensure that the schema has been overridden.
  // TODO... need to call into the appro
  //WINML_THROW_IF_FAILED(adapter_->OverrideSchemaInferenceFunctions());
  return S_OK;
}

OnnxruntimeEnvironment::OnnxruntimeEnvironment(const OrtApi* ort_api) : ort_env_(nullptr, nullptr) {
  // auto winml_adapter_api = GetWinmlAdapterApi(ort_api_);
  OrtEnv* ort_env = nullptr;
  if (auto status = ort_api->CreateEnvWithCustomLogger(&WinmlOrtLoggingCallback, nullptr, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env)) {
    throw;
  }

  ort_env_ = UniqueOrtEnv(ort_env, ort_api->ReleaseEnv);

  OverrideSchemaInferenceFunctions(ort_api);
}

//void Windows::AI::MachineLearning::CWinMLLogSink::SendProfileEvent(onnxruntime::profiling::EventRecord& eventRecord) const {
//  if (eventRecord.cat == onnxruntime::profiling::EventCategory::NODE_EVENT) {
//    TraceLoggingWrite(
//        winmla::winml_trace_logging_provider,
//        "OnnxRuntimeProfiling",
//        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
//        TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
//        TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
//        TraceLoggingString(onnxruntime::profiling::event_categor_names_[eventRecord.cat], "Category"),
//        TraceLoggingInt64(eventRecord.dur, "Duration (us)"),
//        TraceLoggingInt64(eventRecord.ts, "Time Stamp (us)"),
//        TraceLoggingString(eventRecord.name.c_str(), "Event Name"),
//        TraceLoggingInt32(eventRecord.pid, "Process ID"),
//        TraceLoggingInt32(eventRecord.tid, "Thread ID"),
//        TraceLoggingString(eventRecord.args["op_name"].c_str(), "Operator Name"),
//        TraceLoggingString(eventRecord.args["provider"].c_str(), "Execution Provider"));
//  } else {
//    TraceLoggingWrite(
//        winmla::winml_trace_logging_provider,
//        "OnnxRuntimeProfiling",
//        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
//        TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
//        TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
//        TraceLoggingString(onnxruntime::profiling::event_categor_names_[eventRecord.cat], "Category"),
//        TraceLoggingInt64(eventRecord.dur, "Duration (us)"),
//        TraceLoggingInt64(eventRecord.ts, "Time Stamp (us)"),
//        TraceLoggingString(eventRecord.name.c_str(), "Event Name"),
//        TraceLoggingInt32(eventRecord.pid, "Process ID"),
//        TraceLoggingInt32(eventRecord.tid, "Thread ID"));
//  }
//}