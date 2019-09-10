////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// WinMLTelemetryHelper
//
// Copyright (C) Microsoft Corporation
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "pch.h"

WinMLTelemetryHelper::WinMLTelemetryHelper()
    : m_hProvider(g_hWinMLTraceLoggingProvider)
{
}

WinMLTelemetryHelper::~WinMLTelemetryHelper()
{
}

void WinMLTelemetryHelper::LogDllAttachEvent()
{
    if (!m_TelemetryEnabled)
        return;

    WinMLTraceLoggingWrite(
        m_hProvider,
        "ProcessInfo",
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        // Telemetry info
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TraceLoggingUInt8(WINML_TLM_PROCESS_INFO_SCHEMA_VERSION, "schemaVersion"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
}

void WinMLTelemetryHelper::LogSessionCreation(const std::string& modelname, bool isCpu, LUID adapterLuid)
{
    if (!m_TelemetryEnabled)
        return;

    WinMLTraceLoggingWrite(
        m_hProvider,
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
        TraceLoggingInt32(m_runtimeSessionId, "runtimeSessionId"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
}

void WinMLTelemetryHelper::LogModelCreation(bool fromStream,
    const std::string& author,
    const std::string& name,
    const std::string& domain,
    const std::string& description,
    int64_t version,
    bool bUseFP16,
    const std::unordered_map<std::string, std::string>& modelMetadata)
{
    if (!m_TelemetryEnabled)
        return;

    std::string keyStr;
    std::string valueStr;
    for (auto item : modelMetadata)
    {
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
        m_hProvider,
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
        TraceLoggingInt32(m_runtimeSessionId, "runtimeSessionId"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
}

void WinMLTelemetryHelper::LogRuntimeError(HRESULT hr, PCSTR message, PCSTR file, PCSTR function, int line)
{
    if (!m_TelemetryEnabled)
        return;

    WinMLTraceLoggingWrite(
        m_hProvider,
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
        TraceLoggingInt32(m_runtimeSessionId, "runtimeSessionId"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
}

void WinMLTelemetryHelper::LogRuntimeError(HRESULT hr, std::string message, PCSTR file, PCSTR function, int line)
{
    LogRuntimeError(hr, message.c_str(), file, function, line);
}


// The default behavior is to log the telemetry every Nth time this gets called,
// but the caller can override that and log now by specifying force = true.
void WinMLTelemetryHelper::LogRuntimePerf(Profiler<WINML_RUNTIME_PERF>& profiler, bool force)
{
    if (!m_TelemetryEnabled)
        return;

    // we log the telemetry if one of the following is true:
    // 1. "force" was passed to this method. This happens when the dll is unloaded.
    // 2. This method has been called s_telemetryChunkSize number of times.
    // 3. Telemetry hasn't been logged for at least s_telemetryMaxTimeBetweenLogs number of milliseconds.
    static const unsigned int s_telemetryChunkSize = 10000;

    if (!m_timerStarted)
    {
        RestartTimer();
    }

    // 1. "force" was passed to this method. This happens when the dll is unloaded.
    bool shouldLog = force;

    // 2. This method has been called s_telemetryChunkSize number of times.
    if (!shouldLog && ++m_logCounter >= s_telemetryChunkSize)
    {
        shouldLog = true;
    }

    static const ULONGLONG s_telemetryMaxTimeBetweenLogs = 10 * 60 * 1000; // 10 minutes

    if (!shouldLog && (GetTickCount64() - m_timerStart > s_telemetryMaxTimeBetweenLogs))
    {
        shouldLog = true;
    }

    if (shouldLog)
    {
        WinMLTraceLoggingWrite(
            m_hProvider,
            "RuntimePerf",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
            // Telemetry info
            TraceLoggingUInt8(WINML_TLM_RUNTIME_PERF_VERSION, "schemaVersion"),
            TraceLoggingInt32(m_logCounter, "totalEvalCalls"),
            // Load Model Perf Info
            TraceLoggingStruct(22, "loadModelCounters"),
                TraceLoggingInt32(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetCount(), "count"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetAverage(CounterType::TIMER), "avgTime"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetAverage(CounterType::CPU_USAGE), "avgCpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetAverage(CounterType::PAGE_FAULT_COUNT), "avgPageFaultCount"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetAverage(CounterType::WORKING_SET_USAGE), "avgWorkingSetMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetAverage(CounterType::GPU_USAGE), "avgGpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetAverage(CounterType::GPU_DEDICATED_MEM_USAGE), "avgGpuDedicatedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetAverage(CounterType::GPU_SHARED_MEM_USAGE), "avgGpuSharedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMax(CounterType::TIMER), "maxTime"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMax(CounterType::CPU_USAGE), "maxCpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMax(CounterType::PAGE_FAULT_COUNT), "maxPageFaultCount"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMax(CounterType::WORKING_SET_USAGE), "maxWorkingSetMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMax(CounterType::GPU_USAGE), "maxGpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMax(CounterType::GPU_DEDICATED_MEM_USAGE), "maxGpuDedicatedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMax(CounterType::GPU_SHARED_MEM_USAGE), "maxGpuSharedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMin(CounterType::TIMER), "minTime"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMin(CounterType::CPU_USAGE), "minCpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMin(CounterType::PAGE_FAULT_COUNT), "minPageFaultCount"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMin(CounterType::WORKING_SET_USAGE), "minWorkingSetMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMin(CounterType::GPU_USAGE), "minGpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMin(CounterType::GPU_DEDICATED_MEM_USAGE), "minGpuDedicatedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::LOAD_MODEL].GetMin(CounterType::GPU_SHARED_MEM_USAGE), "minGpuSharedMemory"),
            // Evaluate Model Perf Info
            TraceLoggingStruct(22, "evalModelCounters"),
                TraceLoggingInt32(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetCount(), "count"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetAverage(CounterType::TIMER), "avgTime"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetAverage(CounterType::CPU_USAGE), "avgCpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetAverage(CounterType::PAGE_FAULT_COUNT), "avgPageFaultCount"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetAverage(CounterType::WORKING_SET_USAGE), "avgWorkingSetMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetAverage(CounterType::GPU_USAGE), "avgGpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetAverage(CounterType::GPU_DEDICATED_MEM_USAGE), "avgGpuDedicatedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetAverage(CounterType::GPU_SHARED_MEM_USAGE), "avgGpuSharedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMax(CounterType::TIMER), "maxTime"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMax(CounterType::CPU_USAGE), "maxCpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMax(CounterType::PAGE_FAULT_COUNT), "maxPageFaultCount"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMax(CounterType::WORKING_SET_USAGE), "maxWorkingSetMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMax(CounterType::GPU_USAGE), "maxGpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMax(CounterType::GPU_DEDICATED_MEM_USAGE), "maxGpuDedicatedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMax(CounterType::GPU_SHARED_MEM_USAGE), "maxGpuSharedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMin(CounterType::TIMER), "minTime"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMin(CounterType::CPU_USAGE), "minCpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMin(CounterType::PAGE_FAULT_COUNT), "minPageFaultCount"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMin(CounterType::WORKING_SET_USAGE), "minWorkingSetMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMin(CounterType::GPU_USAGE), "minGpuUsage"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMin(CounterType::GPU_DEDICATED_MEM_USAGE), "minGpuDedicatedMemory"),
                TraceLoggingFloat64(profiler[WINML_RUNTIME_PERF::EVAL_MODEL].GetMin(CounterType::GPU_SHARED_MEM_USAGE), "minGpuSharedMemory"),
            TraceLoggingInt32(m_runtimeSessionId, "runtimeSessionId"),
            TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
        );

        // clear the profiler data
        profiler.Reset(ProfilerType::GPU);
        profiler.Reset(ProfilerType::CPU);
        m_logCounter = 0;
        RestartTimer();
    }
}

bool WinMLTelemetryHelper::IsMeasureSampled()
{
    // If the machine isn't sampled at Measure Level, return false.
    return TraceLoggingProviderEnabled(m_hProvider, WINEVENT_LEVEL_LOG_ALWAYS, MICROSOFT_KEYWORD_MEASURES);
}

void
WinMLTelemetryHelper::LogRegisterOperatorKernel(
    const char* name,
    const char* domain,
    int execution_type
)
{
    if (!m_TelemetryEnabled)
        return;

    WinMLTraceLoggingWrite(
        m_hProvider,
        "RegisterOperatorKernel",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        // Telemetry info
        TraceLoggingUInt8(WINML_TLM_RUNTIME_ERROR_VERSION, "schemaVersion"),
       //op kernel info
        TraceLoggingString(name, "name"),
        TraceLoggingString(domain, "domain"),
        TraceLoggingInt32(execution_type, "executionType"),
        TraceLoggingInt32(m_runtimeSessionId, "runtimeSessionId"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
}

void
WinMLTelemetryHelper::RegisterOperatorSetSchema(
    const char* name,
    uint32_t input_count,
    uint32_t output_count,
    uint32_t type_constraint_count,
    uint32_t attribute_count,
    uint32_t default_attribute_count
)
{
    if (!m_TelemetryEnabled)
        return;

    WinMLTraceLoggingWrite(
        m_hProvider,
        "RegisterOperatorSetSchema",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
        TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
        // Telemetry info
        TraceLoggingUInt8(WINML_TLM_RUNTIME_ERROR_VERSION, "schemaVersion"),
        //op kernel info
        TraceLoggingString(name, "name"),
        TraceLoggingInt32(input_count, "inputCount"), //stats
        TraceLoggingInt32(output_count, "outputCount"),
        TraceLoggingInt32(type_constraint_count, "typeConstraintCount"),
        TraceLoggingInt32(attribute_count, "attributeCount"),
        TraceLoggingInt32(default_attribute_count, "defaultAttributeCount"),
        TraceLoggingInt32(m_runtimeSessionId, "m_runtimeSessionId"),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
    );
}