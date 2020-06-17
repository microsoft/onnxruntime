/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/// \file
/// \brief Implements a logging system with a central IO thread that handles
/// all stringification and IO.
/// \details Log-producing threads only submit lambdas to be executed on the
/// IO thread.
/// All producers and consumers use lock-free operations that guarantee
/// forward progress independent of a) other stalled threads and b) where
/// those threads are stalled.
/// Each thread uses a double-buffering scheme to queue its logs. One buffer
/// is always reserved for writes and the other is reserved for reads.
/// A producing thread sends requests to the IOThread to swap the buffers
/// and the IOThread does the actual read/write swap after it has finished
/// reading the buffer it was working on.

#include "logging.h"

#include <cassert>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <process.h>
#include <windows.h>
#define MLPERF_GET_PID() _getpid()
#else
#include <unistd.h>
#define MLPERF_GET_PID() getpid()
#endif

// Use system-level TID for tracing. This enables correlation with other
// performance tools that are not aware of C++ std::thread::id.
#if defined(__linux__)
#include <sys/syscall.h>
#define MLPERF_GET_TID() syscall(SYS_gettid)
#elif defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#define MLPERF_GET_TID() GetCurrentThreadId()
#elif defined(__APPLE__)
#define MLPERF_GET_TID() std::hash<std::thread::id>{}(std::this_thread::get_id())
#else
// TODO: std::this_thread::id is a class but MLPERF_GET_TID() assigned to
// uint64_t
#define MLPERF_GET_TID() std::this_thread::get_id()
#endif

#include "utils.h"

namespace mlperf {
namespace logging {

namespace {

uintptr_t SwapRequestSlotIsWritableValue(size_t id) {
  // LSB of 1 indicates that this isn't a pointer.
  // MSBs encode the id to detect collisions when a slot in
  // |thread_swap_request_slots_| is reused for a different id and the request
  // for the previous id is very slow.
  return (id << 1) | 0x1;
}

bool SwapRequestSlotIsReadable(uintptr_t value) {
  // Valid pointers will not have their lsb set.
  return (value & 0x1) != 0x1;
}

constexpr size_t kMaxThreadsToLog = 1024;
constexpr std::chrono::milliseconds kLogPollPeriod(10);

/// \brief How many log entries to pre-allocate per thread to help avoid
/// runtime allocation.
constexpr size_t kTlsLogReservedEntryCount = 1024;

constexpr auto kInvalidLatency = std::numeric_limits<QuerySampleLatency>::min();

}  // namespace

const std::string& ArgValueTransform(const bool& value) {
  static const std::string v_true("true");
  static const std::string v_false("false");
  return value ? v_true : v_false;
}

char Bin2Hex(uint8_t four_bits) {
  char number = '0' + four_bits;
  char letter = ('A' - 10) + four_bits;
  return four_bits < 10 ? number : letter;
}

const std::string ArgValueTransform(const LogBinaryAsHexString& value) {
  if (value.data == nullptr) {
    return "\"\"";
  }
  std::string hex;
  hex.reserve(value.data->size() + 2);
  hex.push_back('"');
  for (auto b : *value.data) {
    hex.push_back(Bin2Hex(b >> 4));
    hex.push_back(Bin2Hex(b & 0x0F));
  }
  hex.push_back('"');
  return hex;
}

ChromeTracer::ChromeTracer(std::ostream* out, PerfClock::time_point origin)
    : out_(out), origin_(origin) {
  WriteTraceEventHeader();
}

ChromeTracer::~ChromeTracer() {
  WriteTraceEventFooter();
  out_->flush();
}

void ChromeTracer::WriteTraceEventHeader() {
  // Times and durations are converted from nanoseconds to microseconds, use
  // 3 decimal digits to preserve precision.
  *out_ << std::fixed << std::setprecision(3) << "{\"traceEvents\":[\n";
}

void ChromeTracer::WriteTraceEventFooter() {
  *out_ << "{\"name\":\"LastTrace\"}\n"
        << "],\n"
        << "\"displayTimeUnit\":\"ns\",\n"
        << "\"otherData\":{\n"
        << "\"ts\":" << Micros(origin_.time_since_epoch()).count() << ",\n"
        << "\"version\":\"MLPerf LoadGen v0.5a0\"\n"
        << "}\n"
        << "}\n";
}

void AsyncLog::SetCurrentPidTid(uint64_t pid, uint64_t tid) {
  current_pid_ = pid;
  current_tid_ = tid;
}

void AsyncLog::SetLogFiles(std::ostream* summary, std::ostream* detail,
                           std::ostream* accuracy, bool copy_detail_to_stdout,
                           bool copy_summary_to_stdout,
                           PerfClock::time_point log_origin) {
  std::unique_lock<std::mutex> lock(log_mutex_);
  if (summary_out_ != &std::cerr) {
    std::string warning_summary;
    if (log_warning_count_ == 0) {
      warning_summary = "\nNo warnings encountered during test.\n";
    } else if (log_warning_count_ == 1) {
      warning_summary = "\n1 warning encountered. See detailed log.\n";
    } else if (log_warning_count_ != 0) {
      warning_summary = "\n" + std::to_string(log_warning_count_) +
                        " warnings encountered. See detailed log.\n";
    }

    std::string error_summary;
    if (log_error_count_ == 0) {
      error_summary = "\nNo errors encountered during test.\n";
    } else if (log_error_count_ == 1) {
      error_summary = "\n1 ERROR encountered. See detailed log.\n";
    } else if (log_error_count_ != 0) {
      error_summary = "\n" + std::to_string(log_error_count_) +
                      " ERRORS encountered. See detailed log.\n";
    }

    *summary_out_ << warning_summary << error_summary;
    if (copy_summary_to_stdout_) {
      std::cout << warning_summary << error_summary;
    }
  }
  if (summary_out_) {
    summary_out_->flush();
  }
  if (detail_out_) {
    detail_out_->flush();
  }
  if (accuracy_out_ != &std::cerr) {
    WriteAccuracyFooterLocked();
    accuracy_out_->flush();
  }
  summary_out_ = summary;
  detail_out_ = detail;
  accuracy_out_ = accuracy;
  if (accuracy_out_ != &std::cerr) {
    WriteAccuracyHeaderLocked();
  }
  copy_detail_to_stdout_ = copy_detail_to_stdout;
  copy_summary_to_stdout_ = copy_summary_to_stdout;
  log_origin_ = log_origin;
  log_error_count_ = 0;
  log_warning_count_ = 0;
}

void AsyncLog::StartNewTrace(std::ostream* trace_out,
                             PerfClock::time_point origin) {
  std::unique_lock<std::mutex> lock(trace_mutex_);
  if (trace_out) {
    tracer_ = std::make_unique<ChromeTracer>(trace_out, origin);
  } else {
    tracer_.reset();
  }
}

void AsyncLog::StopTrace() {
  std::unique_lock<std::mutex> lock(trace_mutex_);
  tracer_.reset();
}

void AsyncLog::LogAccuracy(uint64_t seq_id, const QuerySampleIndex qsl_idx,
                           const LogBinaryAsHexString& response) {
  std::unique_lock<std::mutex> lock(log_mutex_);
  if (!accuracy_out_) {
    return;
  }
  *accuracy_out_ << (accuracy_needs_comma_ ? ",\n{ " : "\n{ ");
  LogArgs(accuracy_out_, "seq_id", seq_id, "qsl_idx", qsl_idx, "data",
          response);
  *accuracy_out_ << " }";
  accuracy_needs_comma_ = true;
}

void AsyncLog::Flush() {
  {
    std::unique_lock<std::mutex> lock(log_mutex_);
    if (summary_out_) {
      summary_out_->flush();
    }
    if (detail_out_) {
      detail_out_->flush();
    }
    if (accuracy_out_) {
      accuracy_out_->flush();
    }
  }

  {
    std::unique_lock<std::mutex> lock(trace_mutex_);
    if (tracer_) {
      tracer_->Flush();
    }
  }
}

void AsyncLog::WriteAccuracyHeaderLocked() {
  *accuracy_out_ << "[";
  accuracy_needs_comma_ = false;
}

void AsyncLog::WriteAccuracyFooterLocked() { *accuracy_out_ << "\n]\n"; }

void AsyncLog::RestartLatencyRecording(uint64_t first_sample_sequence_id,
                                       size_t latencies_to_reserve) {
  std::unique_lock<std::mutex> lock(latencies_mutex_);
  assert(latencies_.empty());
  assert(latencies_recorded_ == latencies_expected_);
  latencies_recorded_ = 0;
  latencies_expected_ = 0;
  max_latency_ = 0;
  max_completion_timstamp_ = PerfClock::now();
  latencies_first_sample_sequence_id_ = first_sample_sequence_id;
  latencies_.reserve(latencies_to_reserve);
}

void AsyncLog::RecordSampleCompletion(uint64_t sample_sequence_id,
                                      PerfClock::time_point completion_time,
                                      QuerySampleLatency latency) {
  // Relaxed memory order since the early-out checks are inherently racy.
  // The final check will be ordered by locks on the latencies_mutex.
  max_latency_.store(
      std::max(max_latency_.load(std::memory_order_relaxed), latency),
      std::memory_order_relaxed);

  std::unique_lock<std::mutex> lock(latencies_mutex_);

  max_completion_timstamp_ =
      std::max(max_completion_timstamp_, completion_time);

  if (sample_sequence_id < latencies_first_sample_sequence_id_) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync(
        "Received completion for an old sample.", "Min expected id",
        latencies_first_sample_sequence_id_, "Actual id", sample_sequence_id);
    return;
  }

  const size_t i = sample_sequence_id - latencies_first_sample_sequence_id_;

  if (latencies_.size() <= i) {
    // TODO: Reserve in advance.
    latencies_.resize(i + 1, kInvalidLatency);
  } else if (latencies_[i] != kInvalidLatency) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync("Attempted to complete a sample twice.");

    // Return without recording the latency again to avoid potentially
    // ending the test before the SUT is actually done, which could result
    // in a segfault.
    // If the SUT recorded the wrong sample, the test will hang and see
    // the error above.
    return;
  }

  latencies_[i] = latency;
  latencies_recorded_++;
  if (AllLatenciesRecorded()) {
    all_latencies_recorded_.notify_all();
  }
}

std::vector<QuerySampleLatency> AsyncLog::GetLatenciesBlocking(
    size_t expected_count) {
  std::vector<QuerySampleLatency> latencies;
  {
    std::unique_lock<std::mutex> lock(latencies_mutex_);
    latencies_expected_ = expected_count;
    all_latencies_recorded_.wait(lock, [&] { return AllLatenciesRecorded(); });
    latencies.swap(latencies_);
  }

  if (latencies.size() != expected_count) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync("Received SequenceId that was too large.",
                                "expected_size", expected_count, "actual_size",
                                latencies.size());
  }

  size_t invalid_latency_count = 0;
  for (auto l : latencies) {
    if (l == kInvalidLatency) {
      invalid_latency_count++;
    }
  }
  if (invalid_latency_count != 0) {
    // Call LogErrorSync here since this kind of error could result in a
    // segfault in the near future.
    GlobalLogger().LogErrorSync(
        "Encountered incomplete samples at the end of a series of queries.",
        "count", invalid_latency_count);
  }

  return latencies;
}

PerfClock::time_point AsyncLog::GetMaxCompletionTime() {
  return max_completion_timstamp_;
}

QuerySampleLatency AsyncLog::GetMaxLatencySoFar() {
  return max_latency_.load(std::memory_order_relaxed);
}

/// \brief Records a single thread using thread-local storage and submits
/// entries to the central Logger.
///
/// \details This setup allows for each log entry to be added:
///   * With forward-progress guarantees. (i.e.: no locking or blocking
///       operations even if other threads have stalled.)
///   * Without expensive syscalls or I/O operations, which are deferred to
///       the central Logger.
class TlsLogger {
 public:
  TlsLogger(std::function<void()> forced_detatch);
  ~TlsLogger();
  void ForcedDetatchFromThread() { forced_detatch_(); }

  void Log(AsyncLogEntry&& entry);
  void SwapBuffers();

  std::vector<AsyncLogEntry>* StartReadingEntries();
  void FinishReadingEntries();
  bool ReadBufferHasBeenConsumed();
  size_t MaxEntryVectorSize() { return max_entry_size_; }

  uint64_t Pid() const { return pid_; }
  uint64_t Tid() const { return tid_; }

  void RequestSwapBuffersSlotRetried() {
    swap_buffers_slot_retry_count_.fetch_add(1, std::memory_order_relaxed);
  }

  size_t ReportLogCasFailCount() {
    size_t c = log_cas_fail_count_.load(std::memory_order_relaxed);
    log_cas_fail_count_.fetch_sub(c, std::memory_order_relaxed);
    return c;
  }

  size_t ReportSwapBuffersSlotRetryCount() {
    size_t c = swap_buffers_slot_retry_count_.load(std::memory_order_relaxed);
    swap_buffers_slot_retry_count_.fetch_sub(c, std::memory_order_relaxed);
    return c;
  }

  void TraceCounters();

 private:
  using EntryVector = std::vector<AsyncLogEntry>;
  enum class EntryState { Unlocked, ReadLock, WriteLock };

  // Accessed by producer only.
  size_t i_read_ = 0;

  // Accessed by producer and consumer atomically.
  EntryVector entries_[2];
  std::atomic<EntryState> entry_states_[2]{{EntryState::ReadLock},
                                           {EntryState::Unlocked}};
  std::atomic<size_t> i_write_{1};

  std::atomic<size_t> log_cas_fail_count_{0};
  std::atomic<size_t> swap_buffers_slot_retry_count_{0};

  // Accessed by consumer only.
  size_t unread_swaps_ = 0;
  size_t i_write_prev_ = 0;
  uint64_t pid_;
  uint64_t tid_;
  size_t max_entry_size_ = kTlsLogReservedEntryCount;

  std::function<void()> forced_detatch_;
};

Logger::Logger(std::chrono::duration<double> poll_period,
               size_t max_threads_to_log)
    : poll_period_(poll_period),
      max_threads_to_log_(max_threads_to_log),
      thread_swap_request_slots_(max_threads_to_log * 2) {
  const size_t kSlotCount = max_threads_to_log * 2;
  for (size_t i = 0; i < kSlotCount; i++) {
    std::atomic_init(&thread_swap_request_slots_[i],
                     SwapRequestSlotIsWritableValue(i));
  }
}

Logger::~Logger() {
  // TlsLoggers might outlive this Logger when loaded as a python module.
  // Forcefully make all currently registered TlsLoggers orphans.
  std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
  TlsLogger* tls_logger_prev = nullptr;
  (void)tls_logger_prev;  // Avoid unused error in release builds.
  while (!tls_loggers_registerd_.empty()) {
    TlsLogger* tls_logger = *tls_loggers_registerd_.begin();
    // Otherwise, this is an infinite loop.
    assert(tls_logger != tls_logger_prev);
    tls_loggers_registerd_mutex_.unlock();
    tls_logger->ForcedDetatchFromThread();
    tls_loggers_registerd_mutex_.lock();
    tls_logger_prev = tls_logger;
  }
}

void Logger::RequestSwapBuffers(TlsLogger* tls_logger) {
  auto tls_logger_as_uint = reinterpret_cast<uintptr_t>(tls_logger);
  assert(SwapRequestSlotIsReadable(tls_logger_as_uint));
  size_t id, slot;
  uintptr_t slot_is_writeable_value;
  // The compare_exchange below should almost always succeed.
  // The compare_exchange may fail if a recycled slot is still actively used
  // by another thread, so we retry with subsequent slots here if needed.
  // Since the slot count is 2x the expected number of threads to log,
  // the CAS should only fail at most 50% of the time when all logging threads
  // happen to be descheduled between the fetch_add and CAS below, which is
  // very unlikely.
  id = swap_request_id_.fetch_add(1, std::memory_order_relaxed);
  slot = id % thread_swap_request_slots_.size();
  slot_is_writeable_value = SwapRequestSlotIsWritableValue(id);
  while (!thread_swap_request_slots_[slot].compare_exchange_strong(
      slot_is_writeable_value, tls_logger_as_uint, std::memory_order_release)) {
    id = swap_request_id_.fetch_add(1, std::memory_order_relaxed);
    slot = id % thread_swap_request_slots_.size();
    slot_is_writeable_value = SwapRequestSlotIsWritableValue(id);
    tls_logger->RequestSwapBuffersSlotRetried();
  }
}

void Logger::RegisterTlsLogger(TlsLogger* tls_logger) {
  std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
  if (tls_loggers_registerd_.size() >= max_threads_to_log_) {
    LogErrorSync(
        "Warning: More TLS loggers registerd than can "
        "be active simultaneously.\n");
  }
  tls_loggers_registerd_.insert(tls_logger);
}

// This moves ownership of the tls_logger data to Logger so the
// exiting thread can exit immediately, even if all the logs of the
// exiting thread haven't been processed.
void Logger::UnRegisterTlsLogger(std::unique_ptr<TlsLogger> tls_logger) {
  OrphanContainer::iterator orphan;
  {
    std::unique_lock<std::mutex> lock(tls_logger_orphans_mutex_);
    tls_logger_orphans_.emplace_front(std::move(tls_logger));
    orphan = tls_logger_orphans_.begin();
  }

  // Only remove the TlsLogger from the registry after adding to orphans so
  // CollectTlsLoggerStats doesn't have any gaps in coverage.
  {
    std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
    tls_loggers_registerd_.erase(orphan->get());
  }

  // This will flush the logs of |tls_logger| and mark it for destruction.
  // Deferring destruction via orphans_to_destroy helps avoid use-after-frees
  // when the IOThread calls FinishReadingEntries.
  (*orphan)->Log([this, orphan](AsyncLog&) {
    CollectTlsLoggerStats(orphan->get());
    orphans_to_destroy_.push_back(orphan);
  });
}

void Logger::CollectTlsLoggerStats(TlsLogger* tls_logger) {
  tls_total_log_cas_fail_count_ += tls_logger->ReportLogCasFailCount();
  tls_total_swap_buffers_slot_retry_count_ +=
      tls_logger->ReportSwapBuffersSlotRetryCount();

  size_t max_entry_vector_size = tls_logger->MaxEntryVectorSize();
  if (max_entry_vector_size > kTlsLogReservedEntryCount) {
    async_logger_.FlagWarning();
    async_logger_.LogDetail("Logging allocation detected: ", "tid",
                            tls_logger->Tid(), "reserved_entries",
                            kTlsLogReservedEntryCount, "max_entries",
                            max_entry_vector_size);
  }
}

void Logger::StartIOThread() {
  {
    std::unique_lock<std::mutex> lock(io_thread_mutex_);
    keep_io_thread_alive_ = true;
  }
  io_thread_ = std::thread(&Logger::IOThread, this);
}

void Logger::StopIOThread() {
  {
    std::unique_lock<std::mutex> lock(io_thread_mutex_);
    keep_io_thread_alive_ = false;
    io_thread_cv_.notify_all();
  }
  io_thread_.join();
}

void Logger::StartLogging(std::ostream* summary, std::ostream* detail,
                          std::ostream* accuracy, bool copy_detail_to_stdout,
                          bool copy_summary_to_stdout) {
  async_logger_.SetLogFiles(summary, detail, accuracy, copy_detail_to_stdout,
                            copy_summary_to_stdout, PerfClock::now());
}

void Logger::StopLogging() {
  if (std::this_thread::get_id() == io_thread_.get_id()) {
    LogErrorSync("StopLogging() not supported from IO thread.");
    return;
  }

  // Flush logs from this thread.
  std::promise<void> io_thread_flushed_this_thread;
  Log([&](AsyncLog&) { io_thread_flushed_this_thread.set_value(); });
  io_thread_flushed_this_thread.get_future().wait();
  async_logger_.SetLogFiles(&std::cerr, &std::cerr, &std::cerr, false, false,
                            PerfClock::now());
}

void Logger::StartNewTrace(std::ostream* trace_out,
                           PerfClock::time_point origin) {
  async_logger_.StartNewTrace(trace_out, origin);
}

void Logger::StopTracing() {
  // Flush traces from this thread.
  std::promise<void> io_thread_flushed_this_thread;
  Log([&](AsyncLog&) { io_thread_flushed_this_thread.set_value(); });
  io_thread_flushed_this_thread.get_future().wait();
  async_logger_.StopTrace();
}

void Logger::LogContentionAndAllocations() {
  LogDetail([&](AsyncDetail& detail) {
    {
      std::unique_lock<std::mutex> lock(tls_loggers_registerd_mutex_);
      for (auto tls_logger : tls_loggers_registerd_) {
        CollectTlsLoggerStats(tls_logger);
      }
    }

    {
      std::unique_lock<std::mutex> lock(tls_logger_orphans_mutex_);
      for (auto& orphan : tls_logger_orphans_) {
        CollectTlsLoggerStats(orphan.get());
      }
    }

    detail("Log Contention Counters:");
    detail(std::to_string(swap_request_slots_retry_count_) +
           " : swap_request_slots_retry_count");
    detail(std::to_string(swap_request_slots_retry_retry_count_) +
           " : swap_request_slots_retry_retry_count");
    detail(std::to_string(swap_request_slots_retry_reencounter_count_) +
           " : swap_request_slots_retry_reencounter_count");
    detail(std::to_string(start_reading_entries_retry_count_) +
           " : start_reading_entries_retry_count");
    detail(std::to_string(tls_total_log_cas_fail_count_) +
           " : tls_total_log_cas_fail_count");
    detail(std::to_string(tls_total_swap_buffers_slot_retry_count_) +
           " : tls_total_swap_buffers_slot_retry_count");

    swap_request_slots_retry_count_ = 0;
    swap_request_slots_retry_retry_count_ = 0;
    swap_request_slots_retry_reencounter_count_ = 0;
    start_reading_entries_retry_count_ = 0;
    tls_total_log_cas_fail_count_ = 0;
    tls_total_swap_buffers_slot_retry_count_ = 0;
  });
}

void Logger::RestartLatencyRecording(uint64_t first_sample_sequence_id,
                                     size_t latencies_to_reserve) {
  async_logger_.RestartLatencyRecording(first_sample_sequence_id,
                                        latencies_to_reserve);
}

std::vector<QuerySampleLatency> Logger::GetLatenciesBlocking(
    size_t expected_count) {
  return async_logger_.GetLatenciesBlocking(expected_count);
}

PerfClock::time_point Logger::GetMaxCompletionTime() {
  return async_logger_.GetMaxCompletionTime();
}

QuerySampleLatency Logger::GetMaxLatencySoFar() {
  return async_logger_.GetMaxLatencySoFar();
}

TlsLogger* Logger::GetTlsLoggerThatRequestedSwap(size_t slot, size_t next_id) {
  uintptr_t slot_value = thread_swap_request_slots_[slot].load();
  if (SwapRequestSlotIsReadable(slot_value)) {
    // TODO: Convert this block to a simple write once we are confidient
    // that we don't need to check for success.
    bool success = thread_swap_request_slots_[slot].compare_exchange_strong(
        slot_value, SwapRequestSlotIsWritableValue(next_id));
    if (!success) {
      LogErrorSync("CAS failed.", "line", __LINE__);
      assert(success);
    }
    return reinterpret_cast<TlsLogger*>(slot_value);
  }
  return nullptr;
}

void Logger::GatherRetrySwapRequests(std::vector<TlsLogger*>* threads_to_swap) {
  if (swap_request_slots_to_retry_.empty()) {
    return;
  }

  std::vector<SlotRetry> retry_slots;
  retry_slots.swap(swap_request_slots_to_retry_);
  for (auto& slot_retry : retry_slots) {
    TlsLogger* tls_logger =
        GetTlsLoggerThatRequestedSwap(slot_retry.slot, slot_retry.next_id);
    if (tls_logger) {
      threads_to_swap->push_back(tls_logger);
    } else {
      swap_request_slots_to_retry_.push_back(slot_retry);
      swap_request_slots_retry_retry_count_++;
    }
  }
}

void Logger::GatherNewSwapRequests(std::vector<TlsLogger*>* threads_to_swap) {
  auto swap_request_end = swap_request_id_.load(std::memory_order_acquire);
  for (; swap_request_id_read_ < swap_request_end; swap_request_id_read_++) {
    size_t slot = swap_request_id_read_ % thread_swap_request_slots_.size();
    size_t next_id = swap_request_id_read_ + thread_swap_request_slots_.size();
    TlsLogger* tls_logger = GetTlsLoggerThatRequestedSwap(slot, next_id);
    if (tls_logger) {
      threads_to_swap->push_back(tls_logger);
    } else {
      swap_request_slots_retry_count_++;
      // A thread is in the middle of its call to RequestSwapBuffers.
      // Retry later once it's done.
      auto it = std::find_if(swap_request_slots_to_retry_.begin(),
                             swap_request_slots_to_retry_.end(),
                             [=](SlotRetry& s) { return s.slot == slot; });
      if (it == swap_request_slots_to_retry_.end()) {
        // This is the first time we are retrying the slot.
        swap_request_slots_to_retry_.push_back({slot, next_id});
      } else {
        // Whoa. We've been retrying this slot since the last time it was
        // encountered. Just update the next_id.
        it->next_id = next_id;
        swap_request_slots_retry_reencounter_count_++;
      }
    }
  }
}

void Logger::IOThread() {
  while (keep_io_thread_alive_) {
    auto tracer1 =
        MakeScopedTracer([](AsyncTrace& trace) { trace("IOThreadLoop"); });
    {
      auto tracer2 = MakeScopedTracer([](AsyncTrace& trace) { trace("Wait"); });
      std::unique_lock<std::mutex> lock(io_thread_mutex_);
      io_thread_cv_.wait_for(lock, poll_period_,
                             [&] { return !keep_io_thread_alive_; });
    }

    {
      auto tracer3 =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Gather"); });
      std::vector<TlsLogger*> threads_to_swap;
      threads_to_swap.swap(threads_to_swap_deferred_);
      GatherRetrySwapRequests(&threads_to_swap);
      GatherNewSwapRequests(&threads_to_swap);
      for (TlsLogger* thread : threads_to_swap) {
        if (thread->ReadBufferHasBeenConsumed()) {
          thread->SwapBuffers();
          // After swapping a thread, it's ready to be read.
          threads_to_read_.push_back(thread);
        } else {
          // Don't swap buffers again until we've finish reading the
          // previous swap.
          threads_to_swap_deferred_.push_back(thread);
        }
      }
    }

    {
      auto tracer4 =
          MakeScopedTracer([](AsyncTrace& trace) { trace("Process"); });
      // Read from the threads we are confident have activity.
      for (std::vector<TlsLogger*>::iterator thread = threads_to_read_.begin();
           thread != threads_to_read_.end(); thread++) {
        auto tracer5 =
            MakeScopedTracer([tid = (*thread)->Tid()](AsyncTrace& trace) {
              trace("Thread", "tid", tid);
            });
        std::vector<AsyncLogEntry>* entries = (*thread)->StartReadingEntries();
        if (!entries) {
          start_reading_entries_retry_count_++;
          continue;
        }

        async_logger_.SetCurrentPidTid((*thread)->Pid(), (*thread)->Tid());
        for (auto& entry : *entries) {
          // Execute the entry to perform the serialization and I/O.
          entry(async_logger_);
        }
        (*thread)->FinishReadingEntries();
        // Mark for removal by the call to RemoveValue below.
        *thread = nullptr;
      }

      // Only remove threads where reading succeeded so we retry the failed
      // threads the next time around.
      RemoveValue(&threads_to_read_, nullptr);
    }

    // Explicitly flush every time we wake up. The goal being minimization
    // of large implicit flushes which could affect tail latency measurements,
    // especially at percentiles closer to 100%.
    /// \todo Determine if explicitly flushing logs every wake up is better
    /// than relying on implicit flushing.
    {
      auto tracer6 =
          MakeScopedTracer([](AsyncTrace& trace) { trace("FlushAll"); });
      async_logger_.Flush();
    }

    if (!orphans_to_destroy_.empty()) {
      auto tracer7 = MakeScopedTracer(
          [](AsyncTrace& trace) { trace("Abandoning Orphans"); });
      std::unique_lock<std::mutex> lock(tls_logger_orphans_mutex_);
      for (auto orphan : orphans_to_destroy_) {
        tls_logger_orphans_.erase(orphan);
      }
      orphans_to_destroy_.clear();
    }
  }
}

TlsLogger::TlsLogger(std::function<void()> forced_detatch)
    : pid_(MLPERF_GET_PID()),
      tid_(MLPERF_GET_TID()),
      forced_detatch_(std::move(forced_detatch)) {
  for (auto& entry : entries_) {
    entry.reserve(kTlsLogReservedEntryCount);
  }
}

TlsLogger::~TlsLogger() {}

// Log always makes forward progress since it can unconditionally obtain a
// "lock" on at least one of the buffers for writing.
// Notificiation is also lock free.
void TlsLogger::Log(AsyncLogEntry&& entry) {
  size_t cas_fail_count = 0;
  auto unlocked = EntryState::Unlocked;
  size_t i_write = i_write_.load(std::memory_order_relaxed);
  while (!entry_states_[i_write].compare_exchange_strong(
      unlocked, EntryState::WriteLock, std::memory_order_acquire,
      std::memory_order_relaxed)) {
    unlocked = EntryState::Unlocked;
    i_write ^= 1;
    // We may need to try 3 times, since there could be a race with a
    // previous SwapBuffers request and we use memory_order_relaxed when
    // loading i_write_ above.
    cas_fail_count++;
    if (cas_fail_count >= 3) {
      GlobalLogger().LogErrorSync("CAS failed.", "times", cas_fail_count,
                                  "line", __LINE__);
    }
    log_cas_fail_count_.fetch_add(1, std::memory_order_relaxed);
  }
  entries_[i_write].emplace_back(std::forward<AsyncLogEntry>(entry));

  // TODO: Convert this block to a simple write once we are confidient
  // that we don't need to check for success.
  auto write_lock = EntryState::WriteLock;
  bool success = entry_states_[i_write].compare_exchange_strong(
      write_lock, EntryState::Unlocked, std::memory_order_release);
  if (!success) {
    GlobalLogger().LogErrorSync("CAS failed.", "line", __LINE__);
    assert(success);
  }

  bool write_buffer_swapped = i_write_prev_ != i_write;
  if (write_buffer_swapped) {
    GlobalLogger().RequestSwapBuffers(this);
    i_write_prev_ = i_write;
  }
}

void TlsLogger::SwapBuffers() {
  // TODO: Convert this block to a simple write once we are confidient
  // that we don't need to check for success.
  auto read_lock = EntryState::ReadLock;
  bool success = entry_states_[i_read_].compare_exchange_strong(
      read_lock, EntryState::Unlocked, std::memory_order_release);
  if (!success) {
    GlobalLogger().LogErrorSync("CAS failed.", "line", __LINE__);
    assert(success);
  }

  i_write_.store(i_read_, std::memory_order_relaxed);
  i_read_ ^= 1;
  unread_swaps_++;
}

// Returns nullptr if read lock fails.
std::vector<AsyncLogEntry>* TlsLogger::StartReadingEntries() {
  auto unlocked = EntryState::Unlocked;
  if (entry_states_[i_read_].compare_exchange_strong(
          unlocked, EntryState::ReadLock, std::memory_order_acquire,
          std::memory_order_relaxed)) {
    return &entries_[i_read_];
  }
  return nullptr;
}

void TlsLogger::FinishReadingEntries() {
  // Detect first logging allocation and track max allocated size.
  size_t new_size = entries_[i_read_].size();
  if (new_size > max_entry_size_) {
    if (max_entry_size_ == kTlsLogReservedEntryCount) {
      Log([ts = PerfClock::now()](AsyncLog& log) {
        log.TraceAsyncInstant("FirstAllocation", 0, ts);
      });
    }
    max_entry_size_ = new_size;
  }

  entries_[i_read_].clear();
  unread_swaps_--;
}

bool TlsLogger::ReadBufferHasBeenConsumed() { return unread_swaps_ == 0; }

void TlsLogger::TraceCounters() {
  auto tracer = MakeScopedTracer(
      [lcfc = log_cas_fail_count_.load(std::memory_order_relaxed),
       sbsrc = swap_buffers_slot_retry_count_.load(std::memory_order_relaxed)](
          AsyncTrace& trace) {
        trace("TlsLogger:ContentionCounters", "log_cas_fail_count", lcfc,
              "swap_buffers_slot_retry_count", sbsrc);
      });
}

Logger& GlobalLogger() {
  static Logger g_logger(kLogPollPeriod, kMaxThreadsToLog);
  return g_logger;
}

/// \brief Moves ownership of the TlsLogger to Logger on thread exit
/// so no round-trip synchronization with the IO thread is required.
struct TlsLoggerWrapper {
  TlsLoggerWrapper(std::function<void()> forced_detatch)
      : tls_logger(std::make_unique<TlsLogger>(std::move(forced_detatch))) {
    GlobalLogger().RegisterTlsLogger(tls_logger.get());
  }
  ~TlsLoggerWrapper() {
    tls_logger->TraceCounters();
    GlobalLogger().UnRegisterTlsLogger(std::move(tls_logger));
  }
  std::unique_ptr<TlsLogger> tls_logger;
};

TlsLoggerWrapper* InitializeMyTlsLoggerWrapper() {
  thread_local std::unique_ptr<TlsLoggerWrapper> tls_logger_wrapper;
  // forced_detatch lets the global Logger forcefully detatch TlsLoggers
  // from the thread in the Logger's destructor, which may run before
  // thread-local variables are destroyed when the loadgen is used as a python
  // module and dynamically unloaded.
  // Note: We capture a pointer to the tls_logger_wrapper since variables of
  // the thread-local storage class aren't actually captured. C++ spec says
  // only variables of the automatic storage class are captured.
  /// \todo There is a race where the same TlsLoggerWrapper might be
  /// destroyed both naturally and via forced_detatch. Destruction of
  /// the TlsLoggerWrapper should be locked.
  auto forced_detatch = [tls_logger_wrapper = &tls_logger_wrapper]() {
    tls_logger_wrapper->reset();
  };
  tls_logger_wrapper = std::make_unique<TlsLoggerWrapper>(forced_detatch);
  return tls_logger_wrapper.get();
}

TlsLogger* InitializeMyTlsLogger() {
  thread_local TlsLoggerWrapper* wrapper = InitializeMyTlsLoggerWrapper();
  return wrapper->tls_logger.get();
}

void Log(AsyncLogEntry&& entry) {
  thread_local TlsLogger* const tls_logger = InitializeMyTlsLogger();
  tls_logger->Log(std::forward<AsyncLogEntry>(entry));
}

}  // namespace logging
}  // namespace mlperf
