// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <exception>
#include <ctime>
#include <utility>

#include "core/common/exceptions.h"
#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#if defined(__MACH__)
#include <pthread.h>
#else
#include <sys/syscall.h>
#endif
#endif
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace logging {
const char* Category::onnxruntime = "onnxruntime";
const char* Category::System = "System";

using namespace std::chrono;

// GSL_SUPRESS(i.22) is broken. Ignore the warnings for the static local variables that are trivial
// and should not have any destruction order issues via pragmas instead.
// https://developercommunity.visualstudio.com/content/problem/249706/gslsuppress-does-not-work-for-i22-c-core-guideline.html
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(pop)
#pragma warning(disable : 26426)
#endif

static minutes InitLocaltimeOffset(const time_point<system_clock>& epoch) noexcept;

const LoggingManager::Epochs& LoggingManager::GetEpochs() noexcept {
  // we save the value from system clock (which we can convert to a timestamp) as well as the high_resolution_clock.
  // from then on, we use the delta from the high_resolution_clock and apply that to the
  // system clock value.
  static Epochs epochs{high_resolution_clock::now(),
                       system_clock::now(),
                       InitLocaltimeOffset(system_clock::now())};
  return epochs;
}

LoggingManager::LoggingManager(std::unique_ptr<ISink> sink, Severity default_min_severity,
                               bool filter_user_data, int default_max_vlog_level)
    : sink_{std::move(sink)},
      default_min_severity_{default_min_severity},
      default_filter_user_data_{filter_user_data},
      default_max_vlog_level_{default_max_vlog_level} {
  if (sink_ == nullptr) {
    throw std::logic_error("ISink must be provided.");
  }
}

LoggingManager::~LoggingManager() = default;

std::unique_ptr<Logger> LoggingManager::CreateLogger(const std::string& logger_id) {
  return CreateLogger(logger_id, default_min_severity_, default_filter_user_data_, default_max_vlog_level_);
}

std::unique_ptr<Logger> LoggingManager::CreateLogger(const std::string& logger_id,
                                                     const Severity severity,
                                                     bool filter_user_data,
                                                     int vlog_level) {
  auto logger = std::make_unique<Logger>(*this, logger_id, severity, filter_user_data, vlog_level);
  return logger;
}

void LoggingManager::Log(const std::string& logger_id, const Capture& message) const {
  sink_->Send(GetTimestamp(), logger_id, message);
}

void LoggingManager::SendProfileEvent(profiling::EventRecord& eventRecord) const {
  sink_->SendProfileEvent(eventRecord);
}

static minutes InitLocaltimeOffset(const time_point<system_clock>& epoch) noexcept {
  // convert the system_clock time_point (UTC) to localtime and gmtime to calculate the difference.
  // we do this once, and apply that difference in GetTimestamp().
  // NOTE: If we happened to be running over a period where the time changed (e.g. daylight saving started)
  // we won't pickup the change. Not worth the extra cost to be 100% accurate 100% of the time.

  const time_t system_time_t = system_clock::to_time_t(epoch);
  tm local_tm;
  tm utc_tm;

#ifdef _WIN32
  localtime_s(&local_tm, &system_time_t);
  gmtime_s(&utc_tm, &system_time_t);
#else
  localtime_r(&system_time_t, &local_tm);
  gmtime_r(&system_time_t, &utc_tm);
#endif

  const double seconds = difftime(mktime(&local_tm), mktime(&utc_tm));

  // minutes should be accurate enough for timezone conversion
  return minutes{static_cast<int64_t>(seconds / 60)};
}

unsigned int GetThreadId() {
#ifdef _WIN32
  return static_cast<unsigned int>(GetCurrentThreadId());
#elif defined(__MACH__)
  uint64_t tid64;
  pthread_threadid_np(NULL, &tid64);
  return static_cast<unsigned int>(tid64);
#else
  return static_cast<unsigned int>(syscall(SYS_gettid));
#endif
}

//
// Get current process id
//
unsigned int GetProcessId() {
#ifdef _WIN32
  return static_cast<unsigned int>(GetCurrentProcessId());
#elif defined(__MACH__)
  return static_cast<unsigned int>(getpid());
#else
  return static_cast<unsigned int>(syscall(SYS_getpid));
#endif
}

}  // namespace logging
}  // namespace onnxruntime
