// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// uncomment this line or use -DCODEGEN_ENABLE_PROFILER in compiler options to enable profiler events in codegen
// #define CODEGEN_ENABLE_PROFILER

#ifdef CODEGEN_ENABLE_PROFILER
#include "core/common/profiler.h"

namespace onnxruntime {

class ProfilerEvent {
 public:
  ProfilerEvent(const std::string& name) : name_(name) {
    ts_ = profiling::Profiler::Instance().StartTime();
  }

  ~ProfilerEvent() {
    profiling::Profiler::Instance().EndTimeAndRecordEvent(profiling::EventCategory::NODE_EVENT, name_, ts_);
  }

 private:
  TimePoint ts_;
  const std::string name_;
};

}  // namespace onnxruntime

#define CODEGEN_PROFILER_EVENT(name) onnxruntime::ProfilerEvent profiler_event(name)

#else

#define CODEGEN_PROFILER_EVENT(name)

#endif
