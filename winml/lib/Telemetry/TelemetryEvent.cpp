// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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

TelemetryEvent::TelemetryEvent(
    EventCategory category) {
    category_ = category;
    event_id_ = InterlockedIncrement(&s_event_id);

    if (category_ == EventCategory::kModelLoad || 
      category_ == EventCategory::kSessionCreation) {
      WinMLTraceLoggingWrite(
          winml_trace_logging_provider,
          "started event",
          TraceLoggingString(EventCategoryToString(category_), "event"),
          TraceLoggingInt64(event_id_.value(), "eventId"),
          TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
          TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP));
    } else {
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
    if (category_ == EventCategory::kModelLoad ||
        category_ == EventCategory::kSessionCreation) {
    WinMLTraceLoggingWrite(
        winml_trace_logging_provider,
        "stopped event",
        TraceLoggingString(EventCategoryToString(category_), "event"),
        TraceLoggingInt64(event_id_.value(), "eventId"),
        TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP));
    } else {
      WinMLTraceLoggingWrite(
          winml_trace_logging_provider,
          "stopped event",
          TraceLoggingString(EventCategoryToString(category_), "event"),
          TraceLoggingInt64(event_id_.value(), "eventId"),
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_START_STOP));
    }
  }
}
