// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "ep_profiling.h"
#include <array>
#include <chrono>
#include <optional>

// Thread-local profiling state. Tracks the active profiler ID and a per-thread stack of ORT event
// boundaries. Per-thread state is necessary because a single session profiler (single profiler_id)
// can have StartEvent/StopEvent called from multiple threads (e.g., via inter-op parallelism).
struct ThreadLocalProfilingState {
  std::optional<uint64_t> profiler_id;
  std::vector<size_t> ort_event_start_indices;  // Stack of event indices at push time (per-thread)
};
static thread_local ThreadLocalProfilingState tls_profiling_state_;

//
// EpEventManager
//

/*static*/
EpEventManager& EpEventManager::GetInstance() {
  static EpEventManager instance;
  return instance;
}

/*static*/
std::optional<uint64_t> EpEventManager::GetActiveProfilerId() {
  return tls_profiling_state_.profiler_id;
}

uint64_t EpEventManager::RegisterProfiler() {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t result = next_profiler_id_++;

  profiler_state_.insert({result, {}});

  return result;
}

void EpEventManager::UnregisterProfiler(uint64_t profiler_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  profiler_state_.erase(profiler_id);
}

void EpEventManager::PushOrtEvent(uint64_t profiler_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto iter = profiler_state_.find(profiler_id);
  if (iter == profiler_state_.end()) {
    return;
  }

  // Record the current event count in the per-thread stack so we can annotate
  // only this thread's events when PopOrtEvent is called.
  tls_profiling_state_.ort_event_start_indices.push_back(iter->second.events.size());

  // Set the active profiler for this thread so kernels can find it.
  tls_profiling_state_.profiler_id = profiler_id;
}

void EpEventManager::PopOrtEvent(uint64_t profiler_id, const std::string& ort_event_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto iter = profiler_state_.find(profiler_id);
  if (iter == profiler_state_.end() || tls_profiling_state_.ort_event_start_indices.empty()) {
    return;
  }

  size_t start_index = tls_profiling_state_.ort_event_start_indices.back();
  tls_profiling_state_.ort_event_start_indices.pop_back();

  // Annotate this thread's EP events (added since StartEvent) with metadata from the correlated ORT event.
  auto current_thread_id = std::this_thread::get_id();
  for (size_t i = start_index; i < iter->second.events.size(); ++i) {
    Event& ep_event = iter->second.events[i];

    if (ep_event.thread_id == current_thread_id && ep_event.ort_event_name.empty()) {
      ep_event.ort_event_name = ort_event_name;
    }
  }

  // Clear the thread-local when the outermost ORT event on this thread finishes.
  if (tls_profiling_state_.ort_event_start_indices.empty()) {
    tls_profiling_state_.profiler_id.reset();
  }
}

void EpEventManager::AddEpEvent(uint64_t profiler_id, Event event) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto iter = profiler_state_.find(profiler_id);
  if (iter == profiler_state_.end()) {
    return;
  }

  iter->second.events.push_back(std::move(event));
}

void EpEventManager::ConsumeEvents(uint64_t profiler_id, std::vector<Event>& events) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto iter = profiler_state_.find(profiler_id);
  if (iter == profiler_state_.end()) {
    return;
  }

  events.clear();
  std::swap(iter->second.events, events);
}

//
// ExampleKernelEpProfiler
//

static int64_t GetEpClockTimeSinceEpochInNanoseconds() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

ExampleKernelEpProfiler::ExampleKernelEpProfiler(const OrtEpApi& api) : OrtEpProfilerImpl{}, ep_api(api) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  StartProfiling = StartProfilingImpl;
  EndProfiling = EndProfilingImpl;
  StartEvent = StartEventImpl;
  StopEvent = StopEventImpl;

  auto& ep_event_manager = EpEventManager::GetInstance();
  profiler_id = ep_event_manager.RegisterProfiler();

  // Estimate the epoch offset between the ORT and EP profiling clocks to allow converting
  // EP timestamps to ORT timestamps. This example EP happens to use the same clock as ORT, so the offset
  // should ideally be close to zero (and is not really required for this example EP but we show here as an example).
  // Note: based on the computation in GPUTracerManager() in include/onnxruntime/core/common/gpu_profiler_common.h.
  constexpr size_t NUM_EPOCH_OFFSET_MEASUREMENTS = 3;
  int64_t abs_epoch_offset_min = std::numeric_limits<int64_t>::max();

  for (size_t i = 0; i < NUM_EPOCH_OFFSET_MEASUREMENTS; i++) {
    // Each iteration of the loop gets approximately the same time point with both clocks and computes their offset.
    // We take the offset with the smallest absolute value.
    int64_t ort_ts1 = api.GetProfilingClockTimeSinceEpochInNanoseconds();
    int64_t ep_ts = GetEpClockTimeSinceEpochInNanoseconds();
    int64_t ort_ts2 = api.GetProfilingClockTimeSinceEpochInNanoseconds();

    int64_t ort_avg_ts = (ort_ts1 + ort_ts2) / 2;
    int64_t epoch_offset = ort_avg_ts - ep_ts;
    int64_t abs_epoch_offset = std::abs(epoch_offset);

    if (abs_epoch_offset < abs_epoch_offset_min) {
      ep_ort_epoch_offset_ = epoch_offset;
      abs_epoch_offset_min = abs_epoch_offset;
    }
  }
}

ExampleKernelEpProfiler::~ExampleKernelEpProfiler() {
  auto& ep_event_manager = EpEventManager::GetInstance();
  ep_event_manager.UnregisterProfiler(profiler_id);
}

/*static*/
void ORT_API_CALL ExampleKernelEpProfiler::ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
  delete static_cast<ExampleKernelEpProfiler*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::StartProfilingImpl(OrtEpProfilerImpl* /*this_ptr*/,
                                                                    int64_t /*profiling_start_time_ns*/,
                                                                    bool* success_out) noexcept {
  // A more complex EP profiler could do some initialization for profiling utilities (e.g., CPUTI) here.
  *success_out = true;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::StartEventImpl(OrtEpProfilerImpl* this_ptr,
                                                                uint64_t /*ort_event_id*/) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  ep_event_manager.PushOrtEvent(self->profiler_id);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::StopEventImpl(OrtEpProfilerImpl* this_ptr,
                                                               uint64_t /*ort_event_id*/,
                                                               const OrtProfilingEvent* c_ort_event) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  Ort::ConstProfilingEvent ort_event(c_ort_event);
  const char* ort_event_name = ort_event.GetName();

  // Annotate all EP events that were collected during this ORT event with metadata from the ORT event.
  ep_event_manager.PopOrtEvent(self->profiler_id, ort_event_name);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::EndProfilingImpl(
    OrtEpProfilerImpl* this_ptr,
    int64_t profiling_start_time_ns,
    OrtProfilingEventsContainer* c_events_container) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  std::vector<EpEventManager::Event> raw_ep_events;
  ep_event_manager.ConsumeEvents(self->profiler_id, raw_ep_events);

  if (raw_ep_events.empty()) {
    return nullptr;
  }

  std::vector<Ort::ProfilingEvent> events;
  events.reserve(raw_ep_events.size());

  for (EpEventManager::Event& raw_ep_event : raw_ep_events) {
    // First, get the EP event's start time and duration using EP's clock:
    int64_t ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        raw_ep_event.start_time.time_since_epoch())
                        .count();
    int64_t dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         raw_ep_event.end_time - raw_ep_event.start_time)
                         .count();

    // ORT requires event start times (in microseconds) that are relative to ORT's profiling start time.
    // Because an EP may not use a clock with the same epoch as ORT's clock, we add a computed offset to EP timestamps
    // that transforms them to timestamps relative to ORT's profiling clock epoch. This is an approximation.
    int64_t norm_ts_ns = ts_ns + self->ep_ort_epoch_offset_;            // absolute time from ORT's epoch
    int64_t rel_ts_us = (norm_ts_ns - profiling_start_time_ns) / 1000;  // time relative to ORT's profiling start
    int64_t dur_us = dur_ns / 1000;

    // Set parent_name an event arg. The parent_name is just the name of the correlated ORT event.
    std::unordered_map<std::string, std::string> args = {{"parent_name", raw_ep_event.ort_event_name.c_str()}};

    Ort::ProfilingEvent event(OrtProfilingEventCategory_KERNEL, -1, -1, raw_ep_event.name.c_str(),
                              rel_ts_us, dur_us, args);

    events.push_back(std::move(event));
  }

  Ort::UnownedProfilingEventsContainer events_container(c_events_container);
  Ort::Status status = events_container.AddEvents(events);

  return status.release();
  EXCEPTION_TO_RETURNED_STATUS_END
}
