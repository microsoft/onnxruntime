// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "ep_profiling.h"
#include <array>
#include <chrono>
#include <optional>

// Thread-local profiling state. Tracks the active client ID and a per-thread stack of ORT event
// boundaries. Per-thread state is necessary because a single session profiler (single client_id)
// can have StartEvent/StopEvent called from multiple threads (e.g., via inter-op parallelism).
struct ThreadLocalProfilingState {
  std::optional<uint64_t> client_id;
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
std::optional<uint64_t> EpEventManager::GetActiveClientId() {
  return tls_profiling_state_.client_id;
}

uint64_t EpEventManager::RegisterClient() {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t result = next_client_id_++;

  client_state_.insert({result, {}});
  num_clients_++;

  return result;
}

void EpEventManager::UnregisterClient(uint64_t client_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end()) {
    return;
  }

  client_state_.erase(iter);
  --num_clients_;

  if (num_clients_ == 0 && enabled_) {
    Shutdown();
  }
}

void EpEventManager::StartProfiling() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (enabled_) {
    return;
  }

  enabled_ = true;
}

void EpEventManager::PushOrtEvent(uint64_t client_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {
    return;
  }

  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end()) {
    return;
  }

  // Record the current event count in the per-thread stack so we can annotate
  // only this thread's events when PopOrtEvent is called.
  tls_profiling_state_.ort_event_start_indices.push_back(iter->second.events.size());

  // Set the active client for this thread so kernels can find it.
  tls_profiling_state_.client_id = client_id;
}

void EpEventManager::PopOrtEvent(uint64_t client_id, const std::string& ort_event_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {
    return;
  }

  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end() || tls_profiling_state_.ort_event_start_indices.empty()) {
    return;
  }

  size_t start_index = tls_profiling_state_.ort_event_start_indices.back();
  tls_profiling_state_.ort_event_start_indices.pop_back();

  // Annotate this thread's EP events (added since StartEvent) with metadata from the correlated ORT event.
  auto current_thread_id = std::this_thread::get_id();
  for (size_t i = start_index; i < iter->second.events.size(); ++i) {
    if (iter->second.events[i].thread_id == current_thread_id) {
      iter->second.events[i].ort_event_name = ort_event_name;
    }
  }

  // Clear the thread-local when the outermost ORT event on this thread finishes.
  if (tls_profiling_state_.ort_event_start_indices.empty()) {
    tls_profiling_state_.client_id.reset();
  }
}

void EpEventManager::AddEpEvent(uint64_t client_id, Event event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {
    return;
  }

  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end()) {
    return;
  }

  iter->second.events.push_back(std::move(event));
}

void EpEventManager::ConsumeEvents(uint64_t client_id, std::vector<Event>& events) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {
    return;
  }

  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end()) {
    return;
  }

  events.clear();
  std::swap(iter->second.events, events);
}

// Caller should hold mutex_
void EpEventManager::Shutdown() {
  if (!enabled_) {
    return;
  }

  enabled_ = false;
  client_state_.clear();
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
  client_id = ep_event_manager.RegisterClient();
}

ExampleKernelEpProfiler::~ExampleKernelEpProfiler() {
  auto& ep_event_manager = EpEventManager::GetInstance();
  ep_event_manager.UnregisterClient(client_id);
}

/*static*/
void ORT_API_CALL ExampleKernelEpProfiler::ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
  delete static_cast<ExampleKernelEpProfiler*>(this_ptr);
}

/*static*/
bool ORT_API_CALL ExampleKernelEpProfiler::StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                              int64_t profiling_start_time_ns) noexcept {
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();
  ep_event_manager.StartProfiling();

  self->profiling_start_time_ns = profiling_start_time_ns;
  return true;
}

/*static*/
void ORT_API_CALL ExampleKernelEpProfiler::StartEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t /*ort_event_id*/) noexcept {
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  ep_event_manager.PushOrtEvent(self->client_id);
}

/*static*/
void ORT_API_CALL ExampleKernelEpProfiler::StopEventImpl(OrtEpProfilerImpl* this_ptr,
                                                         uint64_t /*ort_event_id*/,
                                                         const OrtProfilingEvent* c_ort_event) noexcept {
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  Ort::ConstProfilingEvent ort_event(c_ort_event);
  const char* ort_event_name = ort_event.GetName();

  // Annotate all EP events that were collected during this ORT event with metadata from the ORT event.
  ep_event_manager.PopOrtEvent(self->client_id, ort_event_name);
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
  ep_event_manager.ConsumeEvents(self->client_id, raw_ep_events);

  if (raw_ep_events.empty()) {
    return nullptr;
  }

  std::vector<Ort::ProfilingEvent> events;
  events.reserve(raw_ep_events.size());

  for (EpEventManager::Event& raw_ep_event : raw_ep_events) {
    // ORT requires timestamps relative to the profiling start time. This example EP uses timestamps with the same
    // epoch as ORT (i.e., Unix epoch), so we can just subtract off `profiling_start_time_ns` from `raw_ep_event.ts_ns`.
    // However, if the EP and ORT do not use timestamps with the same epoch, the EP must convert epochs.
    int64_t ts_us = (raw_ep_event.ts_ns - profiling_start_time_ns) / 1000;
    int64_t dur_us = raw_ep_event.dur_ns / 1000;

    // Set parent_name an event arg. The parent_name is just the name of the correlated ORT event.
    std::unordered_map<std::string, std::string> args = {{"parent_name", raw_ep_event.ort_event_name.c_str()}};

    Ort::ProfilingEvent event(OrtProfilingEventCategory_KERNEL, -1, -1, raw_ep_event.name.c_str(),
                              ts_us, dur_us, args);

    events.push_back(std::move(event));
  }

  Ort::UnownedProfilingEventsContainer events_container(c_events_container);
  Ort::Status status = events_container.AddEvents(events);

  return status.release();
  EXCEPTION_TO_RETURNED_STATUS_END
}
