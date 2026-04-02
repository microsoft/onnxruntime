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

ExampleKernelEpProfiler::ExampleKernelEpProfiler(const OrtEpApi& api) : OrtEpProfilerImpl{}, ep_api(api) {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  StartProfiling = StartProfilingImpl;
  EndProfiling = EndProfilingImpl;
  StartEvent = StartEventImpl;
  StopEvent = StopEventImpl;

  auto& ep_event_manager = EpEventManager::GetInstance();
  profiler_id = ep_event_manager.RegisterProfiler();
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
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                                    int64_t ep_profiling_start_offset_ns) noexcept {
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);

  // Store the offset from ORT's profiling start (measured with ORT's clock) and capture the EP's own clock.
  // This allows computing ORT-relative timestamps without depending on matching clock epochs.
  self->ep_profiling_start_offset_ns_ = ep_profiling_start_offset_ns;
  self->ep_profiling_start_time_point_ = std::chrono::high_resolution_clock::now();
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::StartEventImpl(OrtEpProfilerImpl* this_ptr,
                                                                uint64_t /*ort_event_correlation_id*/) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  ep_event_manager.PushOrtEvent(self->profiler_id);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::StopEventImpl(OrtEpProfilerImpl* this_ptr,
                                                               uint64_t /*ort_event_correlation_id*/,
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
    // ORT requires event timestamps (in microseconds) relative to ORT's profiling start time.
    // We compute this without depending on matching clock epochs by combining:
    //   1. ep_profiling_start_offset_ns_: elapsed time (ORT clock) from ORT's profiling start to StartProfiling call.
    //   2. ep_elapsed_ns: elapsed time (EP clock) from our StartProfiling capture to this event.
    // Their sum is the event's offset from ORT's profiling start.
    int64_t ep_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                raw_ep_event.start_time - self->ep_profiling_start_time_point_)
                                .count();
    int64_t rel_ts_us = (self->ep_profiling_start_offset_ns_ + ep_elapsed_ns) / 1000;

    int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         raw_ep_event.end_time - raw_ep_event.start_time)
                         .count();

    // Set parent_name as an event arg. The parent_name is just the name of the correlated ORT event.
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
