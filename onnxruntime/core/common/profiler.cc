// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "profiler.h"

namespace onnxruntime {
namespace profiling {
using namespace std::chrono;

::onnxruntime::TimePoint profiling::Profiler::StartTime() const {
  return std::chrono::high_resolution_clock::now();
}

void Profiler::StartProfiling(const logging::Logger* session_logger, const std::string& file_name) {
  ONNXRUNTIME_ENFORCE(session_logger != nullptr);
  session_logger_ = session_logger;
  enabled_ = true;
  profile_stream_ = std::ofstream(file_name, std::ios::out | std::ios::trunc);
  profile_stream_file_ = file_name;
  profiling_start_time_ = StartTime();
}

void Profiler::EndTimeAndRecordEvent(EventCategory category,
                                     const std::string& event_name,
                                     TimePoint& start_time,
                                     std::unordered_map<std::string, std::string>&& event_args,
                                     bool /*sync_gpu*/) {
  if (!enabled_)
    return;
  //TODO: sync_gpu if needed.
  std::lock_guard<std::mutex> lock(mutex_);
  if (events_.size() < max_num_events_) {
    long long dur = TimeDiffMicroSeconds(start_time);
    long long ts = TimeDiffMicroSeconds(profiling_start_time_, start_time);
    events_.emplace_back(category, logging::GetProcessId(),
                         logging::GetThreadId(), event_name, ts, dur, std::move(event_args));
  } else {
    if (session_logger_ && !max_events_reached) {
      LOGS(*session_logger_, ERROR)
          << "Maximum number of events reached, could not record profile event.";
      max_events_reached = true;
    }
  }
}

std::string Profiler::WriteProfileData() {
  std::lock_guard<std::mutex> lock(mutex_);
  profile_stream_ << "[\n";

  for (size_t i = 0; i < events_.size(); ++i) {
    auto& rec = events_[i];
    profile_stream_ << R"({"cat" : ")" << event_categor_names_[rec.cat] << "\",";
    profile_stream_ << "\"pid\" :" << rec.pid << ",";
    profile_stream_ << "\"tid\" :" << rec.tid << ",";
    profile_stream_ << "\"dur\" :" << rec.dur << ",";
    profile_stream_ << "\"ts\" :" << rec.ts << ",";
    profile_stream_ << R"("ph" : "X",)";
    profile_stream_ << R"("name" :")" << rec.name << "\",";
    profile_stream_ << "\"args\" : {";
    bool is_first_arg = true;
    for (std::pair<std::string, std::string> event_arg : rec.args) {
      if (!is_first_arg) profile_stream_ << ",";
      profile_stream_ << "\"" << event_arg.first << "\" : \"" << event_arg.second << "\"";
      is_first_arg = false;
    }
    profile_stream_ << "}";
    if (i == events_.size() - 1) {
      profile_stream_ << "}\n";
    } else {
      profile_stream_ << "},\n";
    }
  }
  profile_stream_ << "]\n";
  profile_stream_.close();
  enabled_ = false;  // will not collect profile after writing.
  return profile_stream_file_;
}

//
// Conditionally sync the GPU if the syncGPU flag is set.
//
void ProfilerSyncGpu() {
  ONNXRUNTIME_NOT_IMPLEMENTED("Needs to implement only for gpus");
}

}  // namespace profiling
}  // namespace onnxruntime
