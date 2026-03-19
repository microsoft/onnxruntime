// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../plugin_ep_utils.h"

class EpEventTracer {
 public:
  struct Event {
    std::string name;
    uint64_t ort_event_id;
    int64_t ts_us;
    int64_t dur_us;
  };

  static EpEventTracer& GetInstance() {
    static EpEventTracer instance;
    return instance;
  }

  uint64_t RegisterClient() {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t result = next_client_id_++;

    client_state_.insert({result, {}});
    num_clients_++;

    return result;
  }

  void UnregisterClient(uint64_t client_id) {
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

  void StartProfiling() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enabled_) {
      return;
    }

    enabled_ = true;
  }

  uint64_t PeekOrtEventId(uint64_t client_id) {
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

  void PushOrtEventId(uint64_t client_id, uint64_t ort_event_id) {
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

  void PopOrtEventId(uint64_t client_id) {
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

  void AddEvent(uint64_t client_id, Event event) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = client_state_.find(client_id);

    if (iter == client_state_.end()) {
      return;
    }

    iter->second.events.push_back(std::move(event));
  }

  void ConsumeEvents(uint64_t client_id, std::vector<Event>& events) {
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

 private:
  // Caller should hold mutex_
  void Shutdown() {
    if (!enabled_) {
      return;
    }

    enabled_ = false;
    client_state_.clear();
  }

  struct ClientState {
    std::vector<uint64_t> ort_event_id_stack;
    std::vector<Event> events;
  };

  mutable std::mutex mutex_;
  uint64_t next_client_id_;
  uint64_t num_clients_;
  bool enabled_;

  // client ID -> ClientState
  std::unordered_map<uint64_t, ClientState> client_state_;
};

struct ExampleKernelEpProfiler : OrtEpProfilerImpl {
  const OrtEpApi& ep_api;
  int64_t profiling_start_time_ns = 0;
  uint64_t client_id = 0;

  explicit ExampleKernelEpProfiler(const OrtEpApi& api) : OrtEpProfilerImpl{}, ep_api(api) {
    ort_version_supported = ORT_API_VERSION;
    Release = ReleaseImpl;
    StartProfiling = StartProfilingImpl;
    EndProfiling = EndProfilingImpl;
    StartEvent = StartEventImpl;
    StopEvent = StopEventImpl;

    auto& ep_event_tracer = EpEventTracer::GetInstance();
    client_id = ep_event_tracer.RegisterClient();
  }

  ~ExampleKernelEpProfiler() {
    auto& ep_event_tracer = EpEventTracer::GetInstance();
    ep_event_tracer.UnregisterClient(client_id);
  }

  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
    delete static_cast<ExampleKernelEpProfiler*>(this_ptr);
  }

  static bool ORT_API_CALL StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                              int64_t profiling_start_time_ns) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    auto& ep_event_tracer = EpEventTracer::GetInstance();
    ep_event_tracer.StartProfiling();

    self->profiling_start_time_ns = profiling_start_time_ns;
    return true;
  }

  static void ORT_API_CALL StartEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t ort_event_id) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    auto& ep_event_tracer = EpEventTracer::GetInstance();

    ep_event_tracer.PushOrtEventId(self->client_id, ort_event_id);
  }

  static void ORT_API_CALL StopEventImpl(OrtEpProfilerImpl* this_ptr, uint64_t /*ort_event_id*/) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    auto& ep_event_tracer = EpEventTracer::GetInstance();

    ep_event_tracer.PopOrtEventId(self->client_id);
  }

  static OrtStatus* ORT_API_CALL EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                  int64_t /*profiling_start_time_ns*/,
                                                  OrtEpProfilingEventsContainer* events_container) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    auto& ep_event_tracer = EpEventTracer::GetInstance();

    std::vector<EpEventTracer::Event> raw_ep_events;
    ep_event_tracer.ConsumeEvents(self->client_id, raw_ep_events);

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

    for (EpEventTracer::Event& raw_ep_event : raw_ep_events) {
      OrtEpProfilingEvent* ev = nullptr;

      status = self->ep_api.CreateEpProfilingEvent(
          OrtEpProfilingEventCategory_KERNEL, raw_ep_event.ort_event_id, -1, -1, raw_ep_event.name.c_str(),
          static_cast<int64_t>(raw_ep_event.ts_us), raw_ep_event.dur_us,
          nullptr, nullptr, 0, &ev);

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
};
