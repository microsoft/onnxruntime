#pragma once

#include "core/common/profiler_common.h"
#include "core/common/inlined_containers.h"

#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <utility>


namespace onnxruntime {
namespace profiling {

// The classes in this header are implemented as template/inline classes
// to avoid having to export symbols from the main onnxruntime shared library
// to ExecutionProvider (EP) shared libraries.
// More context: The main onnxruntime shared library is optimized for size
// using --gc-sections during link time to ensure that any unreferenced code
// is not retained. This poses a problem in using a design pattern where the
// (abstract) base class is implemented in the main onnxruntime shared library,
// but (concrete) subclasses are implemented in EP shared libraries. Now, because
// EP shared libraries are loaded at runtime (as of 11/2022), there will be no
// references to the base class symbols when the main onnxruntime shared library
// is compiled. Thus, the base class symbols will not be included in the
// main onnxruntime shared library. This manifests in being unable to load
// EP shared libs (because the base class symbols referenced by derived
// classes are missing).
// We solve this by implementing base classes that are common to all GPU profilers
// inline in this header.

class ProfilerActivityBuffer {
 public:
  ProfilerActivityBuffer()
    : data_(nullptr), size_(0) {}

  ProfilerActivityBuffer(const char* data, size_t size)
    : data_(std::make_unique<char[]>(size)), size_(size) {
    memcpy(data_.get(), data, size_);
  }

  ProfilerActivityBuffer(const ProfilerActivityBuffer& other)
    : ProfilerActivityBuffer(other.GetData(), other.GetSize()) {}

  ProfilerActivityBuffer(ProfilerActivityBuffer&& other)
    : ProfilerActivityBuffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  ProfilerActivityBuffer& operator=(const ProfilerActivityBuffer& other) {
    if (&other == this) {
      return *this;
    }

    new (this) ProfilerActivityBuffer {other};
    return *this;
  }

  ProfilerActivityBuffer& operator=(ProfilerActivityBuffer&& other) {
    if (&other == this) {
      return *this;
    }

    new (this) ProfilerActivityBuffer {std::move(other)};
    return *this;
  }

  static ProfilerActivityBuffer CreateFromPreallocatedBuffer(char* data, size_t size) {
    ProfilerActivityBuffer res{};
    res.data_.reset(data);
    res.size_ = size;
    return res;
  }

  // accessors
  char* GetData() { return data_.get(); }
  const char* GetData() const { return data_.get(); }
  size_t GetSize() const { return size_; }

 private:
  std::unique_ptr<char[]> data_;
  size_t size_;
}; /* end class ProfilerActivityBuffer */

template <typename TDerived>
class GPUTracerManager
{
public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GPUTracerManager);
  virtual ~GPUTracerManager() {}

  uint64_t RegisterClient() {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    auto res = next_client_id_++;
    per_client_events_by_ext_correlation_.insert({res, {}});
    ++num_active_clients_;
    return res;
  }

  void DeregisterClient(uint64_t client_handle) {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    auto it = per_client_events_by_ext_correlation_.find(client_handle);
    if (it == per_client_events_by_ext_correlation_.end()) {
      return;
    }
    per_client_events_by_ext_correlation_.erase(it);
    --num_active_clients_;
    if (num_active_clients_ == 0 && logging_enabled_) {
      StopLogging();
    }
  }

  void StartLogging() {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (logging_enabled_) {
        return;
    }

    auto this_as_derived = static_cast<TDerived*>(this);
    logging_enabled_ = this_as_derived->OnStartLogging();
  }

  void Consume(uint64_t client_handle, const TimePoint& start_time, std::map<uint64_t, Events>& events) {
    auto this_as_derived = static_cast<TDerived*>(this);
    events.clear();
    {
      // Flush any pending activity records before starting
      // to process the accumulated activity records.
      std::lock_guard<std::mutex> lock_manager(manager_instance_mutex_);
      if (!logging_enabled_) {
        return;
      }

      this_as_derived->FlushActivities();
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
      this_as_derived->ProcessActivityBuffers(activity_buffers, start_time);
      auto it = per_client_events_by_ext_correlation_.find(client_handle);
      if (it == per_client_events_by_ext_correlation_.end()) {
        return;
      }
      std::swap(events, it->second);
    }
  }

  bool PushCorrelation(uint64_t client_handle,
                       uint64_t external_correlation_id,
                       TimePoint profiling_start_time) {
    auto this_as_derived = static_cast<TDerived*>(this);
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
    return this_as_derived->PushUniqueCorrelation(unique_cid);
  }

  void PopCorrelation(uint64_t& popped_external_correlation_id) {
    auto this_as_derived = static_cast<TDerived*>(this);
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (!logging_enabled_) {
      return;
    }
    uint64_t unique_cid;
    this_as_derived->PopUniqueCorrelation(unique_cid);
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
  GPUTracerManager() = default;

#if 0
  // Functional API to be implemented by subclasses
  // Included here only for documentation purposes
  bool OnStartLogging();
  void OnStopLogging();
  void ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                              const TimePoint& start_time);
  bool PushUniqueCorrelation(uint64_t unique_cid);
  void PopUniqueCorrelation(uint64_t& popped_unique_cid);
  void FlushActivities();
#endif

  void EnqueueActivityBuffer(ProfilerActivityBuffer&& buffer) {
    std::lock_guard<std::mutex> lock(unprocessed_activity_buffers_mutex_);
    unprocessed_activity_buffers_.emplace_back(std::move(buffer));
  }

  // To be called by subclasses only from ProcessActivityBuffers
  void MapEventToClient(uint64_t tracer_correlation_id, EventRecord&& event) {
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

  // To be called by subclasses only from ProcessActivityBuffers
  void NotifyNewCorrelation(uint64_t tracer_correlation_id, uint64_t unique_correlation_id) {
    tracer_correlation_to_unique_correlation_[tracer_correlation_id] = unique_correlation_id;
    auto pending_it = events_pending_client_mapping_.find(tracer_correlation_id);
    if (pending_it == events_pending_client_mapping_.end()) {
      return;
    }
    // Map the pending events to the right client
    MapEventsToClient(tracer_correlation_id, std::move(pending_it->second));
    events_pending_client_mapping_.erase(pending_it);
  }

private:
  // Requires: manager_instance_mutex_ should be held
  void StopLogging() {
    auto this_as_derived = static_cast<TDerived*>(this);
    if (!logging_enabled_) {
        return;
    }
    this_as_derived->OnStopLogging();
    logging_enabled_ = false;
    Clear();
  }

  // Requires: manager_instance_mutex_ should be held
  void Clear() {
    unprocessed_activity_buffers_.clear();
    unique_correlation_id_to_client_offset_.clear();
    per_client_events_by_ext_correlation_.clear();
    tracer_correlation_to_unique_correlation_.clear();
    events_pending_client_mapping_.clear();
  }

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

// Base class for a GPU profiler
class GPUProfilerBase : public EpProfiler {
protected:
  GPUProfilerBase() = default;
  void MergeEvents(std::map<uint64_t, Events>& events_to_merge, Events& events) {
    Events merged_events;

    auto event_iter = std::make_move_iterator(events.begin());
    auto event_end = std::make_move_iterator(events.end());
    for (auto& map_iter : events_to_merge) {
      auto ts = static_cast<long long>(map_iter.first);
      while (event_iter != event_end && event_iter->ts < ts) {
        merged_events.emplace_back(*event_iter);
        ++event_iter;
      }

      // find the last event with the same timestamp.
      while (event_iter != event_end && event_iter->ts == ts && (event_iter + 1)->ts == ts) {
        ++event_iter;
      }

      if (event_iter != event_end && event_iter->ts == ts) {
        uint64_t increment = 1;
        for (auto& evt : map_iter.second) {
          evt.args["op_name"] = event_iter->args["op_name"];
          evt.args["parent_name"] = event_iter->name;

          // Tracers may not use Jan 1 1970 as an epoch for timestamps.
          // So, we adjust the timestamp here to something sensible.
          evt.ts = event_iter->ts + increment;
          increment += evt.dur;
        }
        merged_events.emplace_back(*event_iter);
        ++event_iter;
      }

      merged_events.insert(merged_events.end(),
                          std::make_move_iterator(map_iter.second.begin()),
                          std::make_move_iterator(map_iter.second.end()));
    }

    // move any remaining events
    merged_events.insert(merged_events.end(), event_iter, event_end);
    std::swap(events, merged_events);
  }
}; /* class GPUProfilerBase */

// Convert a pointer to a hex string
static inline std::string PointerToHexString(const void* ptr) {
  std::ostringstream sstr;
  sstr << std::hex << ptr;
  return sstr.str();
}

} /* end namespace profiling */
} /* end namespace onnxruntime */
