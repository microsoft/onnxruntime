// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

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
      EventCategory eventCategory, bool sendTelemetry);

  ~TelemetryEvent();

 private:
  EventCategory category_;
  std::optional<int64_t> event_id_;
  bool isEvalutionStartEventSend = false;
  bool sendTelemetry_ = true;
};

}  // namespace Windows::AI::MachineLearning::Telemetry