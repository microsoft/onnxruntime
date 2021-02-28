// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// WinMLTelemetryHelper
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "pch.h"

WinMLTelemetryHelper::WinMLTelemetryHelper()
    : provider_(winml_trace_logging_provider) {
}

WinMLTelemetryHelper::~WinMLTelemetryHelper() {
}

void WinMLTelemetryHelper::LogWinMLShutDown() {
  std::string message = BINARY_NAME;
  message += " is unloaded";
  WinMLTraceLoggingWrite(
      provider_,
      "WinMLShutDown",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingString(message.c_str(), "message"));
}

void WinMLTelemetryHelper::LogWinMLSuspended() {
  WinMLTraceLoggingWrite(
      provider_,
      "WinMLSuspended",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"));
}

void WinMLTelemetryHelper::LogRuntimeError(HRESULT hr, PCSTR message, PCSTR file, PCSTR function, int line) {
  if (!telemetry_enabled_)
    return;

  WinMLTraceLoggingWrite(
      provider_,
      "RuntimeError",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
      // Telemetry info
      TraceLoggingUInt8(WINML_TLM_RUNTIME_ERROR_VERSION, "schemaVersion"),
      // Error Info
      TraceLoggingHResult(hr, "hResult"),
      TraceLoggingString(message, "errormessage"),
      TraceLoggingString(file, "file"),
      TraceLoggingString(function, "function"),
      TraceLoggingInt32(line, "line"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
}

void WinMLTelemetryHelper::LogRuntimeError(HRESULT hr, std::string message, PCSTR file, PCSTR function, int line) {
  LogRuntimeError(hr, message.c_str(), file, function, line);
}

bool WinMLTelemetryHelper::IsMeasureSampled() {
  // If the machine isn't sampled at Measure Level, return false.
  return TraceLoggingProviderEnabled(provider_, WINEVENT_LEVEL_LOG_ALWAYS, MICROSOFT_KEYWORD_MEASURES);
}

void WinMLTelemetryHelper::LogRegisterOperatorKernel(
    const char* name,
    const char* domain,
    int execution_type) {
  if (!telemetry_enabled_)
    return;

  WinMLTraceLoggingWrite(
      provider_,
      "RegisterOperatorKernel",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      // Telemetry info
      TraceLoggingUInt8(WINML_TLM_RUNTIME_ERROR_VERSION, "schemaVersion"),
      //op kernel info
      TraceLoggingString(name, "name"),
      TraceLoggingString(domain, "domain"),
      TraceLoggingInt32(execution_type, "executionType"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
}

void WinMLTelemetryHelper::RegisterOperatorSetSchema(
    const char* name,
    uint32_t input_count,
    uint32_t output_count,
    uint32_t type_constraint_count,
    uint32_t attribute_count,
    uint32_t default_attribute_count) {
  if (!telemetry_enabled_)
    return;

  WinMLTraceLoggingWrite(
      provider_,
      "RegisterOperatorSetSchema",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      // Telemetry info
      TraceLoggingUInt8(WINML_TLM_RUNTIME_ERROR_VERSION, "schemaVersion"),
      //op kernel info
      TraceLoggingString(name, "name"),
      TraceLoggingInt32(input_count, "inputCount"),  //stats
      TraceLoggingInt32(output_count, "outputCount"),
      TraceLoggingInt32(type_constraint_count, "typeConstraintCount"),
      TraceLoggingInt32(attribute_count, "attributeCount"),
      TraceLoggingInt32(default_attribute_count, "defaultAttributeCount"),
      TraceLoggingInt32(runtime_session_id_, "runtime_session_id_"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
}

void WinMLTelemetryHelper::SetIntraOpNumThreadsOverride(
    uint32_t num_threads_override) {
  if (!telemetry_enabled_)
    return;
  WinMLTraceLoggingWrite(
      provider_,
      "SetIntraOpNumThreadsOverride",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      //Telemetry info
      TraceLoggingUInt8(WINML_TLM_NATIVE_API_INTRAOP_THREADS_VERSION, "schemaVersion"),
      // num threads info
      TraceLoggingInt32(num_threads_override, "numThreadsOverride"),
      TraceLoggingInt32(std::thread::hardware_concurrency(), "maxThreadsOnMachine"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
}

void WinMLTelemetryHelper::SetNamedDimensionOverride(
    winrt::hstring name, uint32_t value) {
  if (!telemetry_enabled_)
    return;
  WinMLTraceLoggingWrite(
      provider_,
      "SetNamedDimensionOverride",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      //Telemetry info
      TraceLoggingUInt8(WINML_TLM_NAMED_DIMENSION_OVERRIDE_VERSION, "schemaVersion"),
      // named dimension override info
      TraceLoggingWideString(name.c_str(), "dimensionName"),
      TraceLoggingInt32(value, "overrideValue"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
}