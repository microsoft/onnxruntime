// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "ep_profiling.h"

//
// EpEventManager
//

/*static*/
EpEventManager& EpEventManager::GetInstance() {
  static EpEventManager instance;
  return instance;
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

uint64_t EpEventManager::PeekOrtEventId(uint64_t client_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {
    return 0;
  }

  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end() || iter->second.ort_event_id_stack.empty()) {
    return 0;
  }

  return iter->second.ort_event_id_stack.back();
}

void EpEventManager::PushOrtEventId(uint64_t client_id, uint64_t ort_event_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {
    return;
  }

  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end()) {
    return;
  }

  iter->second.ort_event_id_stack.push_back(ort_event_id);
}

void EpEventManager::PopOrtEventId(uint64_t client_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {
    return;
  }

  auto iter = client_state_.find(client_id);
  if (iter == client_state_.end() || iter->second.ort_event_id_stack.empty()) {
    return;
  }

  iter->second.ort_event_id_stack.pop_back();
}

void EpEventManager::AddEpEvent(uint64_t client_id, Event event) {
  std::lock_guard<std::mutex> lock(mutex_);
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
void ORT_API_CALL ExampleKernelEpProfiler::StartEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t ort_event_id) noexcept {
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  ep_event_manager.PushOrtEventId(self->client_id, ort_event_id);
}

/*static*/
void ORT_API_CALL ExampleKernelEpProfiler::StopEventImpl(OrtEpProfilerImpl* this_ptr,
                                                         uint64_t /*ort_event_id*/) noexcept {
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  ep_event_manager.PopOrtEventId(self->client_id);
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEpProfiler::EndProfilingImpl(
    OrtEpProfilerImpl* this_ptr,
    int64_t profiling_start_time_ns,
    OrtEpProfilingEventsContainer* events_container) noexcept {
  auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
  auto& ep_event_manager = EpEventManager::GetInstance();

  std::vector<EpEventManager::Event> raw_ep_events;
  ep_event_manager.ConsumeEvents(self->client_id, raw_ep_events);

  if (raw_ep_events.empty()) {
    return nullptr;
  }

  std::vector<OrtEpProfilingEvent*> events;
  events.reserve(raw_ep_events.size());

  auto cleanup_resources = [&]() {
    for (auto* ev : events) {
      self->ep_api.ReleaseEpProfilingEvent(ev);
    }

    events.clear();
  };

  OrtStatus* status = nullptr;

  for (EpEventManager::Event& raw_ep_event : raw_ep_events) {
    OrtEpProfilingEvent* ev = nullptr;

    // ORT requires timestamps relative to the profiling start time. This example EP uses timestamps with the same
    // epoch as ORT (i.e., Unix epoch), so we can just subtract off `profiling_start_time_ns` from `raw_ep_event.ts_ns`.
    // However, if the EP and ORT do not use timestamps with the same epoch, the EP must convert epochs.
    int64_t ts_us = (raw_ep_event.ts_ns - profiling_start_time_ns) / 1000;
    int64_t dur_us = raw_ep_event.dur_ns / 1000;

    status = self->ep_api.CreateEpProfilingEvent(
        OrtEpProfilingEventCategory_KERNEL, raw_ep_event.ort_event_id, -1, -1, raw_ep_event.name.c_str(),
        ts_us, dur_us, nullptr, nullptr, 0, &ev);

    if (status != nullptr) {
      cleanup_resources();
      return status;
    }
    events.push_back(ev);
  }

  status = self->ep_api.EpProfilingEventsContainer_AddEvents(
      events_container, events.data(), events.size());

  cleanup_resources();
  return status;
}
