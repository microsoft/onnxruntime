// Copyright (c) Microsoft Corporation.
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

void WinMLTelemetryHelper::LogDllAttachEvent() {
  if (!telemetry_enabled_)
    return;

  WinMLTraceLoggingWrite(
      provider_,
      "ProcessInfo",
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      // Telemetry info
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TraceLoggingUInt8(WINML_TLM_PROCESS_INFO_SCHEMA_VERSION, "schemaVersion"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
}

void WinMLTelemetryHelper::LogSessionCreation(const std::string& modelname, bool isCpu, LUID adapterLuid) {
  if (!telemetry_enabled_)
    return;

  WinMLTraceLoggingWrite(
      provider_,
      "SessionCreation",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      // Telemetry info
      TraceLoggingUInt8(WINML_TLM_CONTEXT_CREATION_VERSION, "schemaVersion"),
      // Session info
      TraceLoggingString(modelname.c_str(), "modelname"),
      TraceLoggingBool(isCpu, "isCpu"),
      TraceLoggingUInt32(adapterLuid.LowPart, "adapterLuidLowPart"),
      TraceLoggingUInt32(adapterLuid.HighPart, "adapterLuidHighPart"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
}

void WinMLTelemetryHelper::LogModelCreation(bool fromStream,
                                            const std::string& author,
                                            const std::string& name,
                                            const std::string& domain,
                                            const std::string& description,
                                            int64_t version,
                                            bool bUseFP16,
                                            const std::unordered_map<std::string, std::string>& modelMetadata) {
  if (!telemetry_enabled_)
    return;

  std::string keyStr;
  std::string valueStr;
  for (auto item : modelMetadata) {
    keyStr.append(item.first);
    keyStr.append("|");
    valueStr.append(item.second);
    valueStr.append("|");
  }
  auto BitmapPixelFormatMetadata = modelMetadata.find("Image.BitmapPixelFormat");
  auto ColorSpaceGammaMetadata = modelMetadata.find("Image.ColorSpaceGamma");
  auto NominalPixelRangeMetadata = modelMetadata.find("Image.NominalPixelRange");

  std::string BitmapPixelFormatString = (BitmapPixelFormatMetadata != modelMetadata.end()) ? BitmapPixelFormatMetadata->second : "";
  std::string ColorSpaceGammaString = (ColorSpaceGammaMetadata != modelMetadata.end()) ? ColorSpaceGammaMetadata->second : "";
  std::string NominalPixelRangeString = (NominalPixelRangeMetadata != modelMetadata.end()) ? NominalPixelRangeMetadata->second : "";

  WinMLTraceLoggingWrite(
      provider_,
      "ModelCreation",
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
      // Telemetry info
      TraceLoggingUInt8(WINML_TLM_MODEL_CREATION_VERSION, "schemaVersion"),
      //stream
      TraceLoggingBool(fromStream, "fromStream"),
      // Model Desc
      TraceLoggingString(author.c_str(), "author"),
      TraceLoggingString(name.c_str(), "name"),
      TraceLoggingString(domain.c_str(), "domain"),
      TraceLoggingString(description.c_str(), "description"),
      TraceLoggingInt64(version, "version"),
      TraceLoggingBool(bUseFP16, "usefp16"),
      TraceLoggingString(BitmapPixelFormatString.c_str(), "bitmappixelformat"),
      TraceLoggingString(ColorSpaceGammaString.c_str(), "colorspacegamma"),
      TraceLoggingString(NominalPixelRangeString.c_str(), "nominalpixelrange"),
      // MetaData
      TraceLoggingUInt64(modelMetadata.size(), "metaDataCount"),
      TraceLoggingString(keyStr.c_str(), "metaDataKeys"),
      TraceLoggingString(valueStr.c_str(), "metaDataValues"),
      TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
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

// The default behavior is to log the telemetry every Nth time this gets called,
// but the caller can override that and log now by specifying force = true.
void WinMLTelemetryHelper::LogRuntimePerf(Profiler<WinMLRuntimePerf>& runtime_profiler, bool force) {
  if (!telemetry_enabled_)
    return;

  // we log the telemetry if one of the following is true:
  // 1. "force" was passed to this method. This happens when the dll is unloaded.
  // 2. This method has been called s_telemetryChunkSize number of times.
  // 3. Telemetry hasn't been logged for at least s_telemetryMaxTimeBetweenLogs number of milliseconds.
  static const unsigned int s_telemetryChunkSize = 10000;

  if (!timer_started_) {
    RestartTimer();
  }

  // 1. "force" was passed to this method. This happens when the dll is unloaded.
  bool shouldLog = force;

  // 2. This method has been called s_telemetryChunkSize number of times.
  if (!shouldLog && ++log_counter_ >= s_telemetryChunkSize) {
    shouldLog = true;
  }

  static const ULONGLONG s_telemetryMaxTimeBetweenLogs = 10 * 60 * 1000;  // 10 minutes

  if (!shouldLog && (GetTickCount64() - timer_start_ > s_telemetryMaxTimeBetweenLogs)) {
    shouldLog = true;
  }

  if (shouldLog) {
    WinMLTraceLoggingWrite(
        provider_,
        "RuntimePerf",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
        // Telemetry info
        TraceLoggingUInt8(WINML_TLM_RUNTIME_PERF_VERSION, "schemaVersion"),
        TraceLoggingInt32(log_counter_, "totalEvalCalls"),
        // Load Model Perf Info
        TraceLoggingStruct(22, "loadModelCounters"),
        TraceLoggingInt32(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetCount(), "count"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetAverage(CounterType::TIMER), "avgTime"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetAverage(CounterType::CPU_USAGE), "avgCpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetAverage(CounterType::PAGE_FAULT_COUNT), "avgPageFaultCount"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetAverage(CounterType::WORKING_SET_USAGE), "avgWorkingSetMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetAverage(CounterType::GPU_USAGE), "avgGpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetAverage(CounterType::GPU_DEDICATED_MEM_USAGE), "avgGpuDedicatedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetAverage(CounterType::GPU_SHARED_MEM_USAGE), "avgGpuSharedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMax(CounterType::TIMER), "maxTime"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMax(CounterType::CPU_USAGE), "maxCpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMax(CounterType::PAGE_FAULT_COUNT), "maxPageFaultCount"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMax(CounterType::WORKING_SET_USAGE), "maxWorkingSetMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMax(CounterType::GPU_USAGE), "maxGpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMax(CounterType::GPU_DEDICATED_MEM_USAGE), "maxGpuDedicatedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMax(CounterType::GPU_SHARED_MEM_USAGE), "maxGpuSharedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMin(CounterType::TIMER), "minTime"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMin(CounterType::CPU_USAGE), "minCpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMin(CounterType::PAGE_FAULT_COUNT), "minPageFaultCount"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMin(CounterType::WORKING_SET_USAGE), "minWorkingSetMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMin(CounterType::GPU_USAGE), "minGpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMin(CounterType::GPU_DEDICATED_MEM_USAGE), "minGpuDedicatedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kLoadModel].GetMin(CounterType::GPU_SHARED_MEM_USAGE), "minGpuSharedMemory"),
        // Evaluate Model Perf Info
        TraceLoggingStruct(22, "evalModelCounters"),
        TraceLoggingInt32(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetCount(), "count"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetAverage(CounterType::TIMER), "avgTime"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetAverage(CounterType::CPU_USAGE), "avgCpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetAverage(CounterType::PAGE_FAULT_COUNT), "avgPageFaultCount"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetAverage(CounterType::WORKING_SET_USAGE), "avgWorkingSetMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetAverage(CounterType::GPU_USAGE), "avgGpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetAverage(CounterType::GPU_DEDICATED_MEM_USAGE), "avgGpuDedicatedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetAverage(CounterType::GPU_SHARED_MEM_USAGE), "avgGpuSharedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMax(CounterType::TIMER), "maxTime"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMax(CounterType::CPU_USAGE), "maxCpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMax(CounterType::PAGE_FAULT_COUNT), "maxPageFaultCount"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMax(CounterType::WORKING_SET_USAGE), "maxWorkingSetMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMax(CounterType::GPU_USAGE), "maxGpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMax(CounterType::GPU_DEDICATED_MEM_USAGE), "maxGpuDedicatedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMax(CounterType::GPU_SHARED_MEM_USAGE), "maxGpuSharedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMin(CounterType::TIMER), "minTime"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMin(CounterType::CPU_USAGE), "minCpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMin(CounterType::PAGE_FAULT_COUNT), "minPageFaultCount"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMin(CounterType::WORKING_SET_USAGE), "minWorkingSetMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMin(CounterType::GPU_USAGE), "minGpuUsage"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMin(CounterType::GPU_DEDICATED_MEM_USAGE), "minGpuDedicatedMemory"),
        TraceLoggingFloat64(runtime_profiler[WinMLRuntimePerf::kEvaluateModel].GetMin(CounterType::GPU_SHARED_MEM_USAGE), "minGpuSharedMemory"),
        TraceLoggingInt32(runtime_session_id_, "runtimeSessionId"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));

    // clear the profiler data
    runtime_profiler.Reset(ProfilerType::GPU);
    runtime_profiler.Reset(ProfilerType::CPU);
    log_counter_ = 0;
    RestartTimer();
  }
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