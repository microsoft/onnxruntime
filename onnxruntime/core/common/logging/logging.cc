// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ctime>
#include <exception>
#include <type_traits>
#include <utility>

#include "core/common/exceptions.h"
#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/composite_sink.h"

#ifdef _WIN32
#include <Windows.h>
#include "core/platform/windows/logging/etw_sink.h"
#else
#include <unistd.h>
#if defined(__MACH__) || defined(__wasm__) || defined(_AIX)
#include <pthread.h>
#else
#include <sys/syscall.h>
#endif
#endif

#if __FreeBSD__
#include <sys/thr.h>  // Use thr_self() syscall under FreeBSD to get thread id
#include "logging.h"
#endif

// C++20 has std::chrono::operator<< for std::chrono::system_clock::time_point but we need to check if usage is valid.
// define temporary macro ORT_USE_CXX20_STD_CHRONO to determine whether to use the std::chrono or date implementation.
#define ORT_USE_CXX20_STD_CHRONO __cpp_lib_chrono >= 201803L

// Apply constraints for Apple builds
#if __APPLE__
#include <TargetConditionals.h>

// iOS check must be first as it also has TARGET_OS_MAC set
#if TARGET_OS_IOS
// iOS requires version 16.3
#if (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED < 160300)
#undef ORT_USE_CXX20_STD_CHRONO
#endif

#elif TARGET_OS_MAC
// Xcode added support for C++20's std::chrono::operator<< in SDK version 14.4,
// but the target macOS version must also be >= 13.3 for it to be used.
#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED < 140400) || \
    (defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED < 130300)
#undef ORT_USE_CXX20_STD_CHRONO
#endif

#endif
#endif  // __APPLE__

#if ORT_USE_CXX20_STD_CHRONO
namespace timestamp_stream_insertion_op_ns = std::chrono;
#else
#include "date/date.h"

namespace timestamp_stream_insertion_op_ns = ::date;
#endif

#undef ORT_USE_CXX20_STD_CHRONO

namespace onnxruntime {
namespace logging {

std::ostream& Timestamp::WriteToStream(std::ostream& os) const {
  return timestamp_stream_insertion_op_ns::operator<<(os, time_point_);
}

std::wostream& Timestamp::WriteToWStream(std::wostream& os) const {
  return timestamp_stream_insertion_op_ns::operator<<(os, time_point_);
}

const char* Category::onnxruntime = "onnxruntime";
const char* Category::System = "System";

using namespace std::chrono;

/*
As LoggingManager can be a static, we need to wrap the default instance and mutex in functions
to ensure they're initialized before use in LoggingManager::LoggingManager. If we don't, and
a static LoggingManager is created at startup, the file scope statics here may not have been
initialized.
*/

static std::atomic<void*>& DefaultLoggerManagerInstance() noexcept {
  // this atomic is to protect against attempts to log being made after the default LoggingManager is destroyed.
  // Theoretically this can happen if a Logger instance is still alive and calls Log via its internal
  // pointer to the LoggingManager.
  // As the first thing LoggingManager::Log does is check the static DefaultLoggerManagerInstance() is not null,
  // any further damage should be prevented (in theory).
  static std::atomic<void*> default_instance;
  return default_instance;
}

LoggingManager* LoggingManager::GetDefaultInstance() {
  return static_cast<LoggingManager*>(DefaultLoggerManagerInstance().load());
}

// GSL_SUPPRESS(i.22) is broken. Ignore the warnings for the static local variables that are trivial
// and should not have any destruction order issues via pragmas instead.
// https://developercommunity.visualstudio.com/content/problem/249706/gslsuppress-does-not-work-for-i22-c-core-guideline.html
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26426)
#endif

static std::mutex& DefaultLoggerMutex() noexcept {
  static std::mutex mutex;
  return mutex;
}

Logger* LoggingManager::s_default_logger_ = nullptr;

#ifdef _MSC_VER
#pragma warning(pop)
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

LoggingManager::LoggingManager(std::unique_ptr<ISink> sink, Severity default_min_severity, bool filter_user_data,
                               const InstanceType instance_type, const std::string* default_logger_id,
                               int default_max_vlog_level)
    : sink_{std::move(sink)},
      default_min_severity_{default_min_severity},
      default_filter_user_data_{filter_user_data},
      default_max_vlog_level_{default_max_vlog_level},
      owns_default_logger_{false} {
  if (sink_ == nullptr) {
    ORT_THROW("ISink must be provided.");
  }

#ifndef ORT_MINIMAL_BUILD
  // EnhanceSinkWithEtw() (or tests constructing a CompositeSink with an ETW sink) may have combined the sink with an
  // ETW sink, in which case callers pass the minimum of the configured and ETW severities. Track the ETW severity
  // separately so loggers keep the configured severity.
  if (sink_->GetType() == SinkType::CompositeSink) {
    const auto* composite = static_cast<const CompositeSink*>(sink_.get());
    if (const auto etw_sev = composite->GetMinSeverityOfType(SinkType::EtwSink); etw_sev.has_value()) {
      etw_min_severity_.store(static_cast<int>(*etw_sev), std::memory_order_relaxed);
      if (const auto base_sev = composite->GetMinSeverityOfType(SinkType::BaseSink); base_sev.has_value()) {
        default_min_severity_.store(*base_sev, std::memory_order_relaxed);
      }
    }
  }
#endif  // !ORT_MINIMAL_BUILD

  if (instance_type == InstanceType::Default) {
    if (default_logger_id == nullptr) {
      ORT_THROW("default_logger_id must be provided if instance_type is InstanceType::Default");
    }

    // lock mutex to create instance, and enable logging
    // this matches the mutex usage in Shutdown
    std::lock_guard<std::mutex> guard(DefaultLoggerMutex());

    if (DefaultLoggerManagerInstance().load() != nullptr) {
      ORT_THROW("Only one instance of LoggingManager created with InstanceType::Default can exist at any point in time.");
    }

    // If the following assertion passes, using the atomic to validate calls to Log should
    // be reasonably economical.
    static_assert(std::remove_reference_t<decltype(DefaultLoggerManagerInstance())>::is_always_lock_free);
    DefaultLoggerManagerInstance().store(this);

    CreateDefaultLogger(*default_logger_id);

    owns_default_logger_ = true;
  }
}

LoggingManager::~LoggingManager() {
  if (owns_default_logger_) {
    // lock mutex to reset DefaultLoggerManagerInstance() and free default logger from this instance.
    std::lock_guard<std::mutex> guard(DefaultLoggerMutex());
#if ((__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L)))
    DefaultLoggerManagerInstance().store(nullptr, std::memory_order_release);
#else
    DefaultLoggerManagerInstance().store(nullptr, std::memory_order::memory_order_release);
#endif

    delete s_default_logger_;
    s_default_logger_ = nullptr;
  }
}

void LoggingManager::CreateDefaultLogger(const std::string& logger_id) {
  // this method is only called from ctor in scope where DefaultLoggerMutex() is already locked
  if (s_default_logger_ != nullptr) {
    ORT_THROW("Default logger already set. ");
  }

  s_default_logger_ = CreateLogger(logger_id).release();
}

std::unique_ptr<Logger> LoggingManager::CreateLogger(const std::string& logger_id) {
  return CreateLogger(logger_id, default_min_severity_.load(std::memory_order_relaxed),
                      default_filter_user_data_, default_max_vlog_level_);
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

  // Note: mktime() expects a local time and utc_tm is not a local time.
  // However, we treat utc_tm as a local time in order to compute the local time offset.
  // To avoid DST inconsistency, set utc_tm's tm_isdst to match that of actual local time local_tm.
  utc_tm.tm_isdst = local_tm.tm_isdst;

  const double seconds = difftime(mktime(&local_tm), mktime(&utc_tm));

  // minutes should be accurate enough for timezone conversion
  return minutes{static_cast<int64_t>(seconds / 60)};
}

std::exception LoggingManager::LogFatalAndCreateException(const char* category,
                                                          const CodeLocation& location,
                                                          const char* format_str, ...) {
  std::string exception_msg;

  // create Capture in separate scope so it gets destructed (leading to log output) before we throw.
  {
    ::onnxruntime::logging::Capture c{::onnxruntime::logging::LoggingManager::DefaultLogger(),
                                      ::onnxruntime::logging::Severity::kFATAL, category,
                                      ::onnxruntime::logging::DataType::SYSTEM, location};
    va_list args;
    va_start(args, format_str);

    c.ProcessPrintf(format_str, args);
    va_end(args);

    exception_msg = c.Message();
  }

  return OnnxRuntimeException(location, exception_msg);
}

unsigned int GetThreadId() {
#ifdef _WIN32
  return static_cast<unsigned int>(GetCurrentThreadId());
#elif defined(__MACH__)
  uint64_t tid64;
  pthread_threadid_np(NULL, &tid64);
  return static_cast<unsigned int>(tid64);
#elif __FreeBSD__
  long tid;
  thr_self(&tid);
  return static_cast<unsigned int>(tid);
#elif defined(__wasm__) || defined(_AIX)
  return static_cast<unsigned int>(pthread_self());
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
#elif defined(__MACH__) || defined(__wasm__) || defined(_AIX)
  return static_cast<unsigned int>(getpid());
#else
  return static_cast<unsigned int>(syscall(SYS_getpid));
#endif
}

std::unique_ptr<ISink> EnhanceSinkWithEtw(std::unique_ptr<ISink> existing_sink, logging::Severity original_severity,
                                          logging::Severity etw_severity) {
#ifdef _WIN32
  auto& manager = EtwRegistrationManager::Instance();
  if (manager.IsEnabled()) {
    auto compositeSink = std::make_unique<CompositeSink>();
    compositeSink->AddSink(std::move(existing_sink), original_severity);
    compositeSink->AddSink(std::make_unique<EtwSink>(), etw_severity);
    return compositeSink;
  } else {
    return existing_sink;
  }
#else
  // On non-Windows platforms, just return the existing logger
  (void)original_severity;
  (void)etw_severity;
  return existing_sink;
#endif  // _WIN32
}

Severity OverrideLevelWithEtw(Severity original_severity) {
#ifdef _WIN32
  auto& manager = logging::EtwRegistrationManager::Instance();
  if (manager.IsEnabled() &&
      (manager.Keyword() & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Logs)) != 0) {
    return manager.MapLevelToSeverity();
  }
#endif  // _WIN32
  return original_severity;
}

#ifndef ORT_MINIMAL_BUILD
void LoggingManager::UpdateEtwSeverityFromSinkLocked() noexcept {
  int etw_severity = kNoEtwSinkSeverity;
  if (sink_->GetType() == SinkType::CompositeSink) {
    const auto* composite = static_cast<const CompositeSink*>(sink_.get());
    if (const auto min_etw = composite->GetMinSeverityOfType(SinkType::EtwSink); min_etw.has_value()) {
      etw_severity = static_cast<int>(*min_etw);
    }
  }
  etw_min_severity_.store(etw_severity, std::memory_order_relaxed);
}
#endif  // !ORT_MINIMAL_BUILD

bool LoggingManager::AddSinkOfType(SinkType sink_type, std::function<std::unique_ptr<ISink>()> sinkFactory,
                                   logging::Severity severity) {
  std::lock_guard<std::mutex> guard(sink_mutex_);
  if (sink_->GetType() == sink_type) {
    return false;  // Sink of this type already exists, do not add another
  }
  if (sink_->GetType() == SinkType::CompositeSink &&
      static_cast<const CompositeSink*>(sink_.get())->HasType(sink_type)) {
    return false;  // Sink of this type already exists, do not add another
  }

  auto new_sink = sinkFactory();
  if (new_sink == nullptr) {
    return false;
  }

  if (sink_->GetType() != SinkType::CompositeSink) {
    // Current sink is not a composite, create a new composite sink and add the current sink to it
    auto new_composite = std::make_unique<CompositeSink>();
    new_composite->AddSink(std::move(sink_), default_min_severity_.load(std::memory_order_relaxed));
    sink_ = std::move(new_composite);
  }

  auto* current_composite = static_cast<CompositeSink*>(sink_.get());
  current_composite->AddSink(std::move(new_sink), severity);
#ifndef ORT_MINIMAL_BUILD
  UpdateEtwSeverityFromSinkLocked();
#else
  const Severity updated_severity = std::min(default_min_severity_.load(std::memory_order_relaxed), severity);
  default_min_severity_.store(updated_severity, std::memory_order_relaxed);
  if (s_default_logger_ != nullptr) {
    s_default_logger_->SetSeverity(updated_severity);
  }
#endif
  return true;
}

void LoggingManager::RemoveSink(SinkType sink_type) {
  std::lock_guard<std::mutex> guard(sink_mutex_);

  if (sink_->GetType() == SinkType::CompositeSink) {
    auto* composite_sink = static_cast<CompositeSink*>(sink_.get());
    const Severity new_severity = composite_sink->RemoveSink(sink_type);

    if (composite_sink->HasOnlyOneSink()) {
      // If only one sink remains, replace the CompositeSink with this single sink
      sink_ = composite_sink->GetRemoveSingleSink();
    }

#ifndef ORT_MINIMAL_BUILD
    ORT_UNUSED_PARAMETER(new_severity);
    UpdateEtwSeverityFromSinkLocked();
#else
    default_min_severity_.store(new_severity, std::memory_order_relaxed);
    if (s_default_logger_ != nullptr) {
      s_default_logger_->SetSeverity(new_severity);
    }
#endif
  }
}

void LoggingManager::SetDefaultLoggerSeverity(Severity severity) {
  std::lock_guard<std::mutex> guard(DefaultLoggerMutex());
  if (s_default_logger_ == nullptr) {
    // fail early for attempted misuse. don't use logging macros as we have no logger.
    ORT_THROW("Attempt to use DefaultLogger but none has been registered.");
  }

  auto* default_manager = GetDefaultInstance();
  if (default_manager != nullptr) {
    default_manager->default_min_severity_.store(severity, std::memory_order_relaxed);
  }

  s_default_logger_->SetSeverity(severity);
}

}  // namespace logging
}  // namespace onnxruntime
