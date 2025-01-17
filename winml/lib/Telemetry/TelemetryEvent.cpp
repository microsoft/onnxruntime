// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Telemetry/pch.h"

#include "inc/TelemetryEvent.h"

using namespace _winmlt;

static uint64_t s_event_id = 0;

static const char* EventCategoryToString(EventCategory category) {
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

TelemetryEvent::TelemetryEvent(EventCategory category) {
  auto is_provider_enabled = TraceLoggingProviderEnabled(
    winml_trace_logging_provider, WINEVENT_LEVEL_VERBOSE, WINML_PROVIDER_KEYWORD_START_STOP
  );

  if (is_provider_enabled) {
    category_ = category;
    event_id_ = InterlockedIncrement(&s_event_id);

    WinMLTraceLoggingWrite(
      winml_trace_logging_provider,
      "started event",
      TraceLoggingString(EventCategoryToString(category_), "event"),
      TraceLoggingInt64(event_id_.value(), "eventId"),
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP)
    );
  }
}

TelemetryEvent::~TelemetryEvent() {
  if (event_id_.has_value()) {
    WinMLTraceLoggingWrite(
      winml_trace_logging_provider,
      "stopped event",
      TraceLoggingString(EventCategoryToString(category_), "event"),
      TraceLoggingInt64(event_id_.value(), "eventId"),
      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP)
    );
  }
}
