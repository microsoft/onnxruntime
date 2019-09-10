#pragma once

#include "WinMLProfiler.h"

namespace Windows::AI::MachineLearning::Telemetry
{

enum class EventCategory
{
    ModelLoad = 0,
    SessionCreation,
    Binding,
    Evaluation,
};

class TelemetryEvent
{
public:
    TelemetryEvent(
        EventCategory eventCategory
    );

    ~TelemetryEvent();

private:
    EventCategory category_;
    std::optional<int64_t> event_id_;
};

// Wrapper to telemetry event. if the call throws the destructor is still called
class PerformanceTelemetryEvent : public TelemetryEvent
{
public:
    PerformanceTelemetryEvent(
        WINML_RUNTIME_PERF mode
    );

    ~PerformanceTelemetryEvent();

private:
    WINML_RUNTIME_PERF mode_;
};

}