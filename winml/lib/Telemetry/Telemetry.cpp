#include "pch.h"

WinMLTelemetryHelper g_Telemetry;

TRACELOGGING_DEFINE_PROVIDER(
    g_hWinMLTraceLoggingProvider,
    WINML_PROVIDER_DESC,
    WINML_PROVIDER_GUID,
    TraceLoggingOptionMicrosoftTelemetry()
);

//
// Perf profiling support
//
Profiler<WINML_RUNTIME_PERF> g_Profiler;
