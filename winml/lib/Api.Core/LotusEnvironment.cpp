#include "pch.h"
#include "inc/LotusEnvironment.h"
#include <TraceLoggingProvider.h> 
#include <telemetry\MicrosoftTelemetry.h>
#include <evntrace.h>

bool Windows::AI::MachineLearning::CWinMLLogSink::DebugOutput = false;
void Windows::AI::MachineLearning::CWinMLLogSink::SendImpl(
    const onnxruntime::logging::Timestamp &timestamp,
    const std::string &logger_id,
    const onnxruntime::logging::Capture &message)
{
    //Lotus Fatal and Error Messages are logged as Telemetry, rest are non-telemetry.
    switch(message.Severity())
    {
    case(onnxruntime::logging::Severity::kFATAL): //Telemetry
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "WinMLLogSink",
            TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingLevel(WINEVENT_LEVEL_CRITICAL),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
            TraceLoggingString(message.Category()),
            TraceLoggingUInt32((UINT32)message.Severity()),
            TraceLoggingString(message.Message().c_str()),
            TraceLoggingString(message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str()),
            TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
        );
        break;
    case(onnxruntime::logging::Severity::kERROR): //Telemetry
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "WinMLLogSink",
            TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
            TraceLoggingString(message.Category()),
            TraceLoggingUInt32((UINT32)message.Severity()),
            TraceLoggingString(message.Message().c_str()),
            TraceLoggingString(message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str()),
            TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES)
        );
        break;
    case(onnxruntime::logging::Severity::kWARNING):
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "WinMLLogSink",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingLevel(WINEVENT_LEVEL_WARNING),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
            TraceLoggingString(message.Category()),
            TraceLoggingUInt32((UINT32)message.Severity()),
            TraceLoggingString(message.Message().c_str()),
            TraceLoggingString(message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str())
        );
        break;
    case(onnxruntime::logging::Severity::kINFO):
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "WinMLLogSink",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingLevel(WINEVENT_LEVEL_INFO),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
            TraceLoggingString(message.Category()),
            TraceLoggingUInt32((UINT32)message.Severity()),
            TraceLoggingString(message.Message().c_str()),
            TraceLoggingString(message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str())
        );
        break;
    case(onnxruntime::logging::Severity::kVERBOSE):
        __fallthrough; //Default is Verbose too.
    default:
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "WinMLLogSink",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
            TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
            TraceLoggingString(message.Category()),
            TraceLoggingUInt32((UINT32)message.Severity()),
            TraceLoggingString(message.Message().c_str()),
            TraceLoggingString(message.Location().ToString(onnxruntime::CodeLocation::kFilenameAndPath).c_str())
        );
    }
    if (DebugOutput)
    {
        OutputDebugStringA(std::string(message.Message() + "\r\n").c_str());
    }
}

void Windows::AI::MachineLearning::CWinMLLogSink::SendProfileEvent(onnxruntime::profiling::EventRecord& eventRecord) const
{
    if (eventRecord.cat == onnxruntime::profiling::EventCategory::NODE_EVENT)
    {
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "OnnxRuntimeProfiling",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
            TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
            TraceLoggingString(onnxruntime::profiling::event_categor_names_[eventRecord.cat], "Category"),
            TraceLoggingInt64(eventRecord.dur, "Duration (us)"),
            TraceLoggingInt64(eventRecord.ts, "Time Stamp (us)"),
            TraceLoggingString(eventRecord.name.c_str(), "Event Name"),
            TraceLoggingInt32(eventRecord.pid, "Process ID"),
            TraceLoggingInt32(eventRecord.tid, "Thread ID"),
            TraceLoggingString(eventRecord.args["op_name"].c_str(), "Operator Name"),
            TraceLoggingString(eventRecord.args["provider"].c_str(), "Execution Provider")
        );
    }
    else
    {
        TraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "OnnxRuntimeProfiling",
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
            TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
            TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
            TraceLoggingString(onnxruntime::profiling::event_categor_names_[eventRecord.cat], "Category"),
            TraceLoggingInt64(eventRecord.dur, "Duration (us)"),
            TraceLoggingInt64(eventRecord.ts, "Time Stamp (us)"),
            TraceLoggingString(eventRecord.name.c_str(), "Event Name"),
            TraceLoggingInt32(eventRecord.pid, "Process ID"),
            TraceLoggingInt32(eventRecord.tid, "Thread ID")
        );
    }
}
