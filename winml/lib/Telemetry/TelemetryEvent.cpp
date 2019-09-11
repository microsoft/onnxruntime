#include "pch.h"

#include "inc/TelemetryEvent.h"

using namespace _winmlt;

static uint64_t s_event_id = 0;

static const char*
EventCategoryToString(
    EventCategory category) {
  switch (category) {
    case EventCategory::kModelLoad:
      return "Model load";
    case EventCategory::kSessionCreation:
      return "Session creation";
    case EventCategory::kBinding:
      return "Binding";
    case EventCategory::kEvaluation:
      return "Evaluation";
    default:
      throw std::invalid_argument("category");
  }
}

static EventCategory
GetEventCategoryFromRuntimePerfMode(
    WinMLRuntimePerf mode) {
  switch (mode) {
    case WinMLRuntimePerf::kLoadModel:
      return EventCategory::kModelLoad;
    case WinMLRuntimePerf::kEvaluateModel:
      return EventCategory::kEvaluation;
    default:
      // This should never happen.
      // If caught downstream by cppwinrt this will be converted to
      // a winrt::hresult_invalid_argument(...);
      throw std::invalid_argument("mode");
  }
}

TelemetryEvent::TelemetryEvent(
    EventCategory category) {
  auto is_provider_enabled =
      TraceLoggingProviderEnabled(
          winml_trace_logging_provider,
          WINEVENT_LEVEL_VERBOSE,
          WINML_PROVIDER_KEYWORD_START_STOP);

  if (is_provider_enabled) {
    category_ = category;
    event_id_ = InterlockedIncrement(&s_event_id);

    WinMLTraceLoggingWrite(
        winml_trace_logging_provider,
        "started event",
        TraceLoggingString(EventCategoryToString(category_), "event"),
        TraceLoggingInt64(event_id_.value(), "eventId"),
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP));
  }
}

TelemetryEvent::~TelemetryEvent() {
  if (event_id_.has_value()) {
    WinMLTraceLoggingWrite(
        winml_trace_logging_provider,
        "stopped event",
        TraceLoggingString(EventCategoryToString(category_), "event"),
        TraceLoggingInt64(event_id_.value(), "eventId"),
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP));
  }
}

PerformanceTelemetryEvent::PerformanceTelemetryEvent(
    WinMLRuntimePerf mode) : TelemetryEvent(GetEventCategoryFromRuntimePerfMode(mode)),
                               mode_(mode) {
  WINML_PROFILING_START(profiler, mode_);
}

PerformanceTelemetryEvent::~PerformanceTelemetryEvent() {
  WINML_PROFILING_STOP(profiler, mode_);
  if (mode_ == WinMLRuntimePerf::kEvaluateModel) {
    telemetry_helper.LogRuntimePerf(profiler, false);
  }
}
