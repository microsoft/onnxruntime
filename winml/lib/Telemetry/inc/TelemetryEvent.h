// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace _winml::Telemetry {

enum class EventCategory {
  kModelLoad = 0,
  kSessionCreation,
  kBinding,
  kEvaluation,
};

class TelemetryEvent {
 public:
  TelemetryEvent(EventCategory eventCategory);

  ~TelemetryEvent();

 private:
  EventCategory category_;
  std::optional<int64_t> event_id_;
};

}  // namespace _winml::Telemetry
