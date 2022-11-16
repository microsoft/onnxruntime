// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace profiling {

class ProfilerActivityBuffer {
 public:
  ProfilerActivityBuffer()
      : data_(nullptr), size_(0) {}

  ProfilerActivityBuffer(const char* data, size_t size)
      : data_(std::make_unique<char[]>(size)), size_(size) {
    memcpy(data_.get(), data, size);
  }

  ProfilerActivityBuffer(const ProfilerActivityBuffer& other)
      : ProfilerActivityBuffer(other.data_.get(), other.size_) {}

  ProfilerActivityBuffer(ProfilerActivityBuffer&& other)
      : ProfilerActivityBuffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  ProfilerActivityBuffer& operator=(const ProfilerActivityBuffer& other) {
    if (&other == this) {
      return *this;
    }

    size_ = other.size_;
    data_ = std::make_unique<char[]>(other.size_);
    memcpy(data_.get(), other.data_.get(), size_);
    return *this;
  }

  ProfilerActivityBuffer& operator=(ProfilerActivityBuffer&& other) {
    if (&other == this) {
      return *this;
    }
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    return *this;
  }

  // accessors
  char* GetData() { return data_.get(); }
  const char* GetData() const { return data_.get(); }
  size_t GetSize() const { return size_; }

  static ProfilerActivityBuffer CreateFromPreallocatedBuffer(char* data, size_t size) {
    ProfilerActivityBuffer res{};
    res.data_.reset(data);
    res.size_ = size;
    return res;
  }

 private:
  std::unique_ptr<char[]> data_;
  size_t size_;
};

enum EventCategory {
  SESSION_EVENT = 0,
  NODE_EVENT,
  KERNEL_EVENT,
  API_EVENT,
  EVENT_CATEGORY_MAX
};

// Event descriptions for the above session events.
static constexpr const char* event_categor_names_[EVENT_CATEGORY_MAX] = {
    "Session",
    "Node",
    "Kernel",
    "Api"
};

// Timing record for all events.
struct EventRecord {
  EventRecord() = default;
  EventRecord(EventCategory category,
              int process_id,
              int thread_id,
              std::string&& event_name,
              long long time_stamp,
              long long duration,
              std::unordered_map<std::string, std::string>&& event_args)
      : cat(category),
        pid(process_id),
        tid(thread_id),
        name(std::move(event_name)),
        ts(time_stamp),
        dur(duration),
        args(std::move(event_args)) {}

  EventRecord(EventCategory category,
              int process_id,
              int thread_id,
              const std::string& event_name,
              long long time_stamp,
              long long duration,
              const std::unordered_map<std::string, std::string>& event_args)
      : cat(category),
        pid(process_id),
        tid(thread_id),
        name(event_name),
        ts(time_stamp),
        dur(duration),
        args(event_args) {}

  EventRecord(const EventRecord& other) = default;
  EventRecord(EventRecord&& other) = default;
  EventRecord& operator=(const EventRecord& other) = default;
  EventRecord& operator=(EventRecord&& other) = default;

  EventCategory cat = EventCategory::API_EVENT;
  int pid = -1;
  int tid = -1;
  std::string name{};
  long long ts = 0;
  long long dur = 0;
  std::unordered_map<std::string, std::string> args{};
};

using Events = std::vector<EventRecord>;

class GPUTracerManager
{
public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GPUTracerManager);
  virtual ~GPUTracerManager() {}

  virtual uint64_t RegisterClient() {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (logging_enabled_) {
      auto res = next_client_id_++;
      per_client_events_by_ext_correlation_.insert({res, {}});
      ++num_active_clients_;
      return res;
    }
    return 0;
  }

  virtual void DeregisterClient(uint64_t client_handle) {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (logging_enabled_) {
      auto it = per_client_events_by_ext_correlation_.find(client_handle);
      if (it == per_client_events_by_ext_correlation_.end()) {
        return;
      }
      per_client_events_by_ext_correlation_.erase(it);
      --num_active_clients_;
      if (num_active_clients_ == 0) {
        StopLogging();
      }
    }
  }

  virtual void StartLogging() = 0;
  virtual void Consume(uint64_t client_handle, const TimePoint& start_time, std::map<uint64_t, Events>& events) {
    events.clear();
    {
      // Flush any pending activity records before starting
      // to process the accumulated activity records.
      std::lock_guard<std::mutex> lock_manager(manager_instance_mutex_);
      FlushActivities();
    }

    std::vector<ProfilerActivityBuffer> activity_buffers;
    {
      std::lock_guard<std::mutex> lock(unprocessed_activity_buffers_mutex_);
      std::swap(unprocessed_activity_buffers_, activity_buffers);
      unprocessed_activity_buffers_.clear();
    }

    {
      // Ensure that at most one thread is working through the activity buffers at any time.
      std::lock_guard<std::mutex> lock_two(activity_buffer_processor_mutex_);
      ProcessActivityBuffers(activity_buffers, start_time);
      auto it = per_client_events_by_ext_correlation_.find(client_handle);
      if (it == per_client_events_by_ext_correlation_.end()) {
        return;
      }
      std::swap(events, it->second);
    }
  }

  virtual bool PushCorrelation(uint64_t client_handle,
                               uint64_t external_correlation_id,
                               TimePoint profiling_start_time) {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (!logging_enabled_) {
      return false;
    }

    auto it = per_client_events_by_ext_correlation_.find(client_handle);
    if (it == per_client_events_by_ext_correlation_.end()) {
      // not a registered client, do nothing
      return false;
    }

    // external_correlation_id is simply the timestamp of this event,
    // relative to profiling_start_time. i.e., it was computed as:
    // external_correlation_id =
    //      std::chrono::duration_cast<std::chrono::microseconds>(event_start_time - profiling_start_time).count()
    //
    // Because of the relative nature of the external_correlation_id, the same
    // external_correlation_id can be reused across different clients, which then makes it
    // impossible to recover the client from the external_correlation_id, which in turn
    // makes it impossible to map events (which are tagged with external_correlation_id) to clients.
    //
    // To address these difficulties, we construct a new correlation_id (let's call it unique_cid)
    // as follows:
    // unique_cid =
    //    external_correlation_id +
    //    std::chrono::duration_cast<std::chrono::microseconds>(profiling_start_time.time_since_epoch()).count()
    // now, unique_cid is monotonically increasing with time, so it can be used to reliably map events to clients.
    //
    // Of course, clients expect lists of events to be returned (on a call to Consume()), that are
    // still keyed on the external_correlation_id that they've specified here, so we need to remember the
    // offset to be subtracted
    uint64_t offset = std::chrono::duration_cast<std::chrono::microseconds>(profiling_start_time.time_since_epoch()).count();
    auto unique_cid = external_correlation_id + offset;
    unique_correlation_id_to_client_offset_[unique_cid] = std::make_pair(client_handle, offset);
    return PushUniqueCorrelation(unique_cid);
  }

  virtual void PopCorrelation(uint64_t& popped_external_correlation_id) {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (!logging_enabled_) {
      return;
    }
    uint64_t unique_cid;
    PopUniqueCorrelation(unique_cid);
    // lookup the offset and subtract it before returning popped_external_correlation_id to the client
    auto client_it = unique_correlation_id_to_client_offset_.find(unique_cid);
    if (client_it == unique_correlation_id_to_client_offset_.end()) {
      popped_external_correlation_id = 0;
      return;
    }
    popped_external_correlation_id = unique_cid - client_it->second.second;
  }

  void PopCorrelation() {
    uint64_t unused;
    PopCorrelation(unused);
  }

protected:
  GPUTracerManager() {}

  void EnqueueActivityBuffer(ProfilerActivityBuffer&& buffer) {
    std::lock_guard<std::mutex> lock(unprocessed_activity_buffers_mutex_);
    unprocessed_activity_buffers_.emplace_back(std::move(buffer));
  }

  // Requires: manager_instance_mutex_ must be held
  virtual void Clear() {
    unprocessed_activity_buffers_.clear();
    unique_correlation_id_to_client_offset_.clear();
    per_client_events_by_ext_correlation_.clear();
    tracer_correlation_to_unique_correlation_.clear();
  }

  virtual void StopLogging() = 0;
  virtual void ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                                      const TimePoint& start_time) = 0;

  virtual bool PushUniqueCorrelation(uint64_t unique_cid) = 0;
  virtual void PopUniqueCorrelation(uint64_t& popped_unique_cid) = 0;
  virtual void FlushActivities() = 0;

  Events* GetEventListForUniqueCorrelationId(uint64_t unique_correlation_id) {
    auto client_it = unique_correlation_id_to_client_offset_.find(unique_correlation_id);
    if (client_it == unique_correlation_id_to_client_offset_.end()) {
      return nullptr;
    }

    // See the comments on the GetUniqueCorrelationId method for an explanation of
    // of this offset computation and why it's required.
    auto const& client_handle_offset = client_it->second;
    auto external_correlation = unique_correlation_id - client_handle_offset.second;

    auto& event_list = per_client_events_by_ext_correlation_[client_handle_offset.first][external_correlation];
    return &event_list;
  }

  // Not thread-safe: subclasses must ensure mutual-exclusion when calling this method
  void MapEventToClient(uint64_t tracer_correlation_id, EventRecord&& event)
  {
    auto it = tracer_correlation_to_unique_correlation_.find(tracer_correlation_id);
    if (it == tracer_correlation_to_unique_correlation_.end()) {
      // We're yet to receive a mapping to unique_correlation_id for this tracer_correlation_id
      DeferEventMapping(std::move(event), tracer_correlation_id);
      return;
    }
    auto unique_correlation_id = it->second;
    auto p_event_list = GetEventListForUniqueCorrelationId(unique_correlation_id);
    if (p_event_list != nullptr) {
      p_event_list->emplace_back(std::move(event));
    }
  }

  // Not thread-safe: subclasses must ensure mutual-exclusion when calling this method
  void MapEventsToClient(uint64_t unique_correlation_id, std::vector<EventRecord>&& events) {
    auto p_event_list = GetEventListForUniqueCorrelationId(unique_correlation_id);
    if (p_event_list != nullptr) {
      p_event_list->insert(p_event_list->end(),
                           std::make_move_iterator(events.begin()),
                           std::make_move_iterator(events.end()));
    }
  }

  void DeferEventMapping(EventRecord&& event, uint64_t tracer_correlation_id) {
    events_pending_client_mapping_[tracer_correlation_id].emplace_back(std::move(event));
  }

  void NotifyOnCorrelation(uint64_t tracer_correlation_id, uint64_t unique_correlation_id) {
    tracer_correlation_to_unique_correlation_[tracer_correlation_id] = unique_correlation_id;
    auto pending_it = events_pending_client_mapping_.find(tracer_correlation_id);
    if (pending_it == events_pending_client_mapping_.end()) {
      return;
    }
    // Map the pending events to the right client
    MapEventsToClient(tracer_correlation_id, std::move(pending_it->second));
    events_pending_client_mapping_.erase(pending_it);
  }

  std::mutex manager_instance_mutex_;
  uint64_t next_client_id_ = 1;
  uint64_t num_active_clients_ = 0;
  bool logging_enabled_ = false;
  std::mutex unprocessed_activity_buffers_mutex_;
  std::mutex activity_buffer_processor_mutex_;

  // Unprocessed activity buffers
  std::vector<ProfilerActivityBuffer> unprocessed_activity_buffers_;

  // Keyed on unique_correlation_id -> (client_id/client_handle, offset)
  // unique_correlation_id - offset == external_correlation_id
  InlinedHashMap<uint64_t, std::pair<uint64_t, uint64_t>> unique_correlation_id_to_client_offset_;

  // Keyed on tracer_correlation_id -> unique_correlation_id
  InlinedHashMap<uint64_t, uint64_t> tracer_correlation_to_unique_correlation_;

  // client_id/client_handle -> external_correlation_id -> events
  InlinedHashMap<uint64_t, std::map<uint64_t, Events>> per_client_events_by_ext_correlation_;

  // Keyed on tracer correlation_id, keeps track of activity records
  // for which we haven't established the external_correlation_id yet.
  InlinedHashMap<uint64_t, std::vector<EventRecord>> events_pending_client_mapping_;
}; /* class GPUTracerManager */

//Execution Provider Profiler
class EpProfiler {
 public:
  virtual ~EpProfiler() = default;
  virtual bool StartProfiling(TimePoint profiling_start_time) = 0;      // called when profiling starts
  virtual void EndProfiling(TimePoint start_time, Events& events) = 0;  // called when profiling ends, save all captures numbers to "events"
  virtual void Start(uint64_t){};                                       // called before op start, accept an id as argument to identify the op
  virtual void Stop(uint64_t){};                                        // called after op stop, accept an id as argument to identify the op
};

// Demangle C++ symbols
std::string demangle(const char* name);
std::string demangle(const std::string& name);

// Convert a pointer to a hex string
static inline std::string PointerToHexString(const void* ptr) {
  std::ostringstream sstr;
  sstr << std::hex << ptr;
  return sstr.str();
}

}  // namespace profiling
}  // namespace onnxruntime
