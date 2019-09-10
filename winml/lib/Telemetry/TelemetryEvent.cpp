#include "pch.h"

#include "inc/TelemetryEvent.h"

using namespace _winmlt;

static uint64_t s_event_id = 0;

static
const char*
EventCategoryToString(
    EventCategory category
)
{
    switch (category)
    {
    case EventCategory::ModelLoad:
        return "Model load";
    case EventCategory::SessionCreation:
        return "Session creation";
    case EventCategory::Binding:
        return "Binding";
    case EventCategory::Evaluation:
        return "Evaluation";
    default:
        throw std::invalid_argument("category");
    }
}

static
EventCategory
GetEventCategoryFromRuntimePerfMode(
    WINML_RUNTIME_PERF mode
)
{
    switch (mode)
    {
    case WINML_RUNTIME_PERF::LOAD_MODEL:
        return EventCategory::ModelLoad;
    case WINML_RUNTIME_PERF::EVAL_MODEL:
        return EventCategory::Evaluation;
    default:
        // This should never happen.
        // If caught downstream by cppwinrt this will be converted to
        // a winrt::hresult_invalid_argument(...);
        throw std::invalid_argument("mode");
    }
}

TelemetryEvent::TelemetryEvent(
    EventCategory category
)
{
    auto is_provider_enabled =
        TraceLoggingProviderEnabled(
            g_hWinMLTraceLoggingProvider,
            WINEVENT_LEVEL_VERBOSE,
            WINML_PROVIDER_KEYWORD_START_STOP);

    if (is_provider_enabled)
    {
        category_ = category;
        event_id_ = InterlockedIncrement(&s_event_id);

        WinMLTraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "started event",
            TraceLoggingString(EventCategoryToString(category_), "event"),
            TraceLoggingInt64(event_id_.value(), "eventId"),
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP)
        );
    }
}

TelemetryEvent::~TelemetryEvent()
{
    if (event_id_.has_value())
    {
        WinMLTraceLoggingWrite(
            g_hWinMLTraceLoggingProvider,
            "stopped event",
            TraceLoggingString(EventCategoryToString(category_), "event"),
            TraceLoggingInt64(event_id_.value(), "eventId"),
            TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP)
        );
    }
}

PerformanceTelemetryEvent::PerformanceTelemetryEvent(
    WINML_RUNTIME_PERF mode) :
        TelemetryEvent(GetEventCategoryFromRuntimePerfMode(mode)),
        mode_(mode)
{
    WINML_PROFILING_START(g_Profiler, mode_);
}

PerformanceTelemetryEvent::~PerformanceTelemetryEvent()
{
    WINML_PROFILING_STOP(g_Profiler, mode_);
    if (mode_ == WINML_RUNTIME_PERF::EVAL_MODEL)
    {
        g_Telemetry.LogRuntimePerf(g_Profiler, false);
    }
}
