// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "profiler.h"
#include <cmath>

#ifdef USE_CUDA
#include <cupti.h>
#endif

namespace onnxruntime {
namespace profiling {
using namespace std::chrono;

class DeviceProfiler {
 public:
  static DeviceProfiler* GetDeviceProfiler();
  virtual void StartProfiling(TimePoint start_time, int pid, int tid) = 0;
  virtual std::vector<EventRecord> EndProfiling() = 0;
  virtual ~DeviceProfiler() = default;
};

#ifdef USE_CUDA
#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))
#define DUR(s, e) std::lround(static_cast<double>(e - s) / 1000)

class CudaProfiler final: public DeviceProfiler {
 public:
  friend class DeviceProfiler;
  ~CudaProfiler() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaProfiler);
  void StartProfiling(TimePoint start_time, int pid, int tid) override;
  std::vector<EventRecord> EndProfiling() override;
 private:
  CudaProfiler() = default;
  static void CUPTIAPI BufferRequested(uint8_t**, size_t*, size_t*);
  static void CUPTIAPI BufferCompleted(CUcontext, uint32_t, uint8_t*, size_t, size_t);
  struct KernelStat {
    std::string name_ = {};
    uint32_t stream_ = 0;
    int32_t grid_x_ = 0;
    int32_t grid_y_ = 0;
    int32_t grid_z_ = 0;
    int32_t block_x_ = 0;
    int32_t block_y_ = 0;
    int32_t block_z_ = 0;
    int64_t start_ = 0;
    int64_t stop_ = 0;
  };
  static OrtMutex mutex_;
  static std::vector<KernelStat> stats_;
  bool initialized_ = false;
  TimePoint start_time_;
  int pid_ = 0;
  int tid_ = 0;
  static std::atomic_flag enabled_;
};

OrtMutex CudaProfiler::mutex_;
std::vector<CudaProfiler::KernelStat> CudaProfiler::stats_;
std::atomic_flag CudaProfiler::enabled_;

void CUPTIAPI CudaProfiler::BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  uint8_t* bfr = (uint8_t*)malloc(BUF_SIZE + ALIGN_SIZE);
  ORT_ENFORCE(bfr, "Failed to allocate memory for cuda kernel profiling.");
  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI CudaProfiler::BufferCompleted(CUcontext, uint32_t, uint8_t*, size_t, size_t) {
}

/*
void CUPTIAPI CudaProfiler::BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  CUptiResult status;
  CUpti_Activity* record = NULL;
  if (validSize > 0) {
    std::unique_lock<OrtMutex> lock(mutex_);
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        if (CUPTI_ACTIVITY_KIND_KERNEL == record->kind) {
          CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*)record;
          stats_.push_back({kernel->name, kernel->streamId,
                            kernel->gridX, kernel->gridY, kernel->gridZ,
                            kernel->blockX, kernel->blockY, kernel->blockZ,
                            static_cast<int64_t>(kernel->start),
                            static_cast<int64_t>(kernel->end)});
        }
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
    } while (1);
  }
  free(buffer);
}*/

void CudaProfiler::StartProfiling(TimePoint start_time, int pid, int tid) {
  if (!enabled_.test_and_set()) {
    start_time_ = start_time;
    pid_ = pid;
    tid_ = tid;
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL) == CUPTI_SUCCESS &&
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted) == CUPTI_SUCCESS) {
      initialized_ = true;
    }
  }
}

std::vector<EventRecord> CudaProfiler::EndProfiling() {
  std::vector<EventRecord> events;
  if (enabled_.test_and_set()) {
    if (initialized_) {
      // cuptiActivityFlushAll(1);
      std::unique_lock<OrtMutex> lock(mutex_);
      int64_t profiling_start = std::chrono::duration_cast<nanoseconds>(start_time_.time_since_epoch()).count();
      for (const auto& stat : stats_) {
        std::initializer_list<std::pair<std::string, std::string>> args = {{"stream", std::to_string(stat.stream_)},
                                                                           {"grid_x", std::to_string(stat.grid_x_)},
                                                                           {"grid_y", std::to_string(stat.grid_y_)},
                                                                           {"grid_z", std::to_string(stat.grid_z_)},
                                                                           {"block_x", std::to_string(stat.block_x_)},
                                                                           {"block_y", std::to_string(stat.block_y_)},
                                                                           {"block_z", std::to_string(stat.block_z_)}};
        events.push_back({EventCategory::KERNEL_EVENT, pid_, tid_, stat.name_, DUR(profiling_start, stat.stop_), DUR(stat.start_, stat.stop_), {args.begin(), args.end()}});
      }
      stats_.clear();
    } else {
      std::initializer_list<std::pair<std::string, std::string>> args;
      events.push_back({EventCategory::KERNEL_EVENT, pid_, tid_, "not_available_due_to_cupti_error", 0, 0, {args.begin(), args.end()}});
    }
  }
  enabled_.clear();
  return events;
}

#endif //USE_CUDA

DeviceProfiler* DeviceProfiler::GetDeviceProfiler() {
#ifdef USE_CUDA
  static CudaProfiler cuda_profiler;
  return &cuda_profiler;
#else
  return nullptr;
#endif
}

std::atomic<size_t> Profiler::global_max_num_events_{1000 * 1000};

#ifdef ENABLE_STATIC_PROFILER_INSTANCE
Profiler* Profiler::instance_ = nullptr;

profiling::Profiler::~Profiler() {
  instance_ = nullptr;
}
#else
profiling::Profiler::~Profiler() {
}
#endif

::onnxruntime::TimePoint profiling::Profiler::Now() const {
  ORT_ENFORCE(enabled_);
  return std::chrono::high_resolution_clock::now();
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
  profiling_start_time_ = Now();
  DeviceProfiler* device_profiler = DeviceProfiler::GetDeviceProfiler();
  if (device_profiler) {
    device_profiler->StartProfiling(profiling_start_time_, logging::GetProcessId(), logging::GetThreadId());
  }
}

template <typename T>
void Profiler::StartProfiling(const std::basic_string<T>& file_name) {
  enabled_ = true;
  profile_stream_.open(file_name, std::ios::out | std::ios::trunc);
  profile_stream_file_ = ToMBString(file_name);
  profiling_start_time_ = Now();
  DeviceProfiler* device_profiler = DeviceProfiler::GetDeviceProfiler();
  if (device_profiler) {
    device_profiler->StartProfiling(profiling_start_time_, logging::GetProcessId(), logging::GetThreadId());
  }
}

template void Profiler::StartProfiling<char>(const std::basic_string<char>& file_name);
#ifdef _WIN32
template void Profiler::StartProfiling<wchar_t>(const std::basic_string<wchar_t>& file_name);
#endif

void Profiler::EndTimeAndRecordEvent(EventCategory category,
                                     const std::string& event_name,
                                     const TimePoint& start_time, const TimePoint& end_time,
                                     const std::initializer_list<std::pair<std::string, std::string>>& event_args,
                                     bool sync_gpu) {
  EndTimeAndRecordEvent(category, event_name, TimeDiffMicroSeconds(start_time, end_time),
                        TimeDiffMicroSeconds(profiling_start_time_, start_time), event_args, sync_gpu);
}

void Profiler::EndTimeAndRecordEvent(EventCategory category,
                                     const std::string& event_name,
                                     const TimePoint& start_time,
                                     const std::initializer_list<std::pair<std::string, std::string>>& event_args,
                                     bool sync_gpu) {
  EndTimeAndRecordEvent(category, event_name, TimeDiffMicroSeconds(start_time),
                        TimeDiffMicroSeconds(profiling_start_time_, start_time), event_args, sync_gpu);
}

void Profiler::EndTimeAndRecordEvent(EventCategory category,
                                     const std::string& event_name,
                                     long long duration,         //duration of the op
                                     long long time_from_start,  //time difference between op start time and profiler start time
                                     const std::initializer_list<std::pair<std::string, std::string>>& event_args,
                                     bool /*sync_gpu*/) {
  EventRecord event(category, logging::GetProcessId(),
                    logging::GetThreadId(), event_name, time_from_start, duration, {event_args.begin(), event_args.end()});
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

  DeviceProfiler* device_profiler = DeviceProfiler::GetDeviceProfiler();
  if (device_profiler) {
    std::vector<EventRecord> device_events = device_profiler->EndProfiling();
    std::copy(device_events.begin(), device_events.end(), std::back_inserter(events_));
  }

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
