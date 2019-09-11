// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "WinMLProfiler.h"

namespace Windows::AI::MachineLearning::Telemetry {

enum class EventCategory {
  kModelLoad = 0,
  kSessionCreation,
  kBinding,
  kEvaluation,
};

class TelemetryEvent {
 public:
  TelemetryEvent(
      EventCategory eventCategory);

  ~TelemetryEvent();

 private:
  EventCategory category_;
  std::optional<int64_t> event_id_;
};

// Wrapper to telemetry event. if the call throws the destructor is still called
class PerformanceTelemetryEvent : public TelemetryEvent {
 public:
  PerformanceTelemetryEvent(
      WinMLRuntimePerf mode);

  ~PerformanceTelemetryEvent();

 private:
  WinMLRuntimePerf mode_;
};

}  // namespace Windows::AI::MachineLearning::Telemetry