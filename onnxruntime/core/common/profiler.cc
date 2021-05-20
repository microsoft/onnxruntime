// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "profiler.h"

namespace onnxruntime {
namespace profiling {
using namespace std::chrono;

std::atomic<size_t> Profiler::global_max_num_events_{1000 * 1000};

#ifdef ENABLE_STATIC_PROFILER_INSTANCE
Profiler* Profiler::instance_ = nullptr;

profiling::Profiler::~Profiler() {
  instance_ = nullptr;
}
#else
profiling::Profiler::~Profiler() {}
#endif

::onnxruntime::TimePoint profiling::Profiler::StartTime() const {
  ORT_ENFORCE(enabled_);
  return std::chrono::steady_clock::now();
}

void Profiler::Initialize(const logging::Logger* session_logger) {
  ORT_ENFORCE(session_logger != nullptr);
  session_logger_ = session_logger;
#ifdef ENABLE_STATIC_PROFILER_INSTANCE
  // In current design, profiler instance goes with inference session. Since it's possible to have
  // multiple inference sessions, profiler by definition is not singleton. However, in performance
  // debugging, it would be helpful to access profiler in code that have no access to inference session,
  // which is why we have this pseudo-singleton implementation here for debugging in single inference session.
  ORT_ENFORCE(instance_ == nullptr, "Static profiler instance only works with single session");
  instance_ = this;
#endif
}

void Profiler::StartProfiling(const logging::Logger* custom_logger) {
  ORT_ENFORCE(custom_logger != nullptr);
  enabled_ = true;
  profile_with_logger_ = true;
  custom_logger_ = custom_logger;
  profiling_start_time_ = StartTime();
}

template <typename T>
void Profiler::StartProfiling(const std::basic_string<T>& file_name) {
  enabled_ = true;
  profile_stream_.open(file_name, std::ios::out | std::ios::trunc);
  profile_stream_file_ = ToMBString(file_name);
  profiling_start_time_ = StartTime();
}

template void Profiler::StartProfiling<char>(const std::basic_string<char>& file_name);
#ifdef _WIN32
template void Profiler::StartProfiling<wchar_t>(const std::basic_string<wchar_t>& file_name);
#endif

void Profiler::EndTimeAndRecordEvent(EventCategory category,
                                     const std::string& event_name,
                                     const TimePoint& start_time,
                                     const std::initializer_list<std::pair<std::string, std::string>>& event_args,
                                     bool /*sync_gpu*/) {
  long long dur = TimeDiffMicroSeconds(start_time);
  long long ts = TimeDiffMicroSeconds(profiling_start_time_, start_time);

  EventRecord event(category, logging::GetProcessId(),
                    logging::GetThreadId(), event_name, ts, dur, {event_args.begin(), event_args.end()});
  if (profile_with_logger_) {
    custom_logger_->SendProfileEvent(event);
  } else {
    //TODO: sync_gpu if needed.
    std::lock_guard<OrtMutex> lock(mutex_);
    if (events_.size() < max_num_events_) {
      events_.emplace_back(event);
    } else {
      if (session_logger_ && !max_events_reached) {
        LOGS(*session_logger_, ERROR)
            << "Maximum number of events reached, could not record profile event.";
        max_events_reached = true;
      }
    }
  }
}

std::string Profiler::EndProfiling() {
  if (!enabled_) {
    return std::string();
  }
  if (profile_with_logger_) {
    profile_with_logger_ = false;
    return std::string();
  }

  if (session_logger_) {
    LOGS(*session_logger_, INFO) << "Writing profiler data to file " << profile_stream_file_;
  }

  std::lock_guard<OrtMutex> lock(mutex_);
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
      if (!event_arg.second.empty() && event_arg.second[0] == '{') {
        profile_stream_ << "\"" << event_arg.first << "\" : " << event_arg.second << "";
      } else {
        profile_stream_ << "\"" << event_arg.first << "\" : \"" << event_arg.second << "\"";
      }
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

}  // namespace profiling
}  // namespace onnxruntime
