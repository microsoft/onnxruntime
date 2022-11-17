#include "core/common/gpu_profiler_common.h"

namespace onnxruntime {
namespace profiling {

// Implementation of ProfilerActivityBuffer
ProfilerActivityBuffer::ProfilerActivityBuffer()
    : data_(nullptr), size_(0) {}

ProfilerActivityBuffer::ProfilerActivityBuffer(const char* data, size_t size)
    : data_(std::make_unique<char[]>(size)), size_(size) {
    memcpy(data_.get(), data, size);
}

ProfilerActivityBuffer::ProfilerActivityBuffer(const ProfilerActivityBuffer& other)
    : ProfilerActivityBuffer(other.data_.get(), other.size_) {}

ProfilerActivityBuffer::ProfilerActivityBuffer(ProfilerActivityBuffer&& other)
    : ProfilerActivityBuffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
}

ProfilerActivityBuffer& ProfilerActivityBuffer::operator=(const ProfilerActivityBuffer& other) {
    if (&other == this) {
      return *this;
    }

    size_ = other.size_;
    data_ = std::make_unique<char[]>(other.size_);
    memcpy(data_.get(), other.data_.get(), size_);
    return *this;
}

ProfilerActivityBuffer& ProfilerActivityBuffer::operator=(ProfilerActivityBuffer&& other) {
    if (&other == this) {
      return *this;
    }
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    return *this;
}

ProfilerActivityBuffer ProfilerActivityBuffer::CreateFromPreallocatedBuffer(char* data, size_t size) {
    ProfilerActivityBuffer res{};
    res.data_.reset(data);
    res.size_ = size;
    return res;
}


// Implementation of GPUTracerManager
GPUTracerManager::~GPUTracerManager() {}

uint64_t GPUTracerManager::RegisterClient() {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (logging_enabled_) {
      auto res = next_client_id_++;
      per_client_events_by_ext_correlation_.insert({res, {}});
      ++num_active_clients_;
      return res;
    }
    return 0;
}

void GPUTracerManager::DeregisterClient(uint64_t client_handle) {
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

void GPUTracerManager::StartLogging() {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (logging_enabled_) {
        return;
    }

    logging_enabled_ = OnStartLogging();
}

void GPUTracerManager::StopLogging() {
    std::lock_guard<std::mutex> lock(manager_instance_mutex_);
    if (!logging_enabled_) {
        return;
    }
    OnStopLogging();
    logging_enabled_ = false;
    Clear();
}

void GPUTracerManager::Consume(uint64_t client_handle,
                               const TimePoint& start_time,
                               std::map<uint64_t, Events>& events) {
    events.clear();
    {
      // Flush any pending activity records before starting
      // to process the accumulated activity records.
      std::lock_guard<std::mutex> lock_manager(manager_instance_mutex_);
      if (!logging_enabled_) {
        return;
      }

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

bool GPUTracerManager::PushCorrelation(uint64_t client_handle,
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

void GPUTracerManager::PopCorrelation(uint64_t& popped_external_correlation_id) {
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

void GPUTracerManager::PopCorrelation() {
    uint64_t unused;
    PopCorrelation(unused);
}

void GPUTracerManager::EnqueueActivityBuffer(ProfilerActivityBuffer&& buffer) {
    std::lock_guard<std::mutex> lock(unprocessed_activity_buffers_mutex_);
    unprocessed_activity_buffers_.emplace_back(std::move(buffer));
}

void GPUTracerManager::Clear() {
    unprocessed_activity_buffers_.clear();
    unique_correlation_id_to_client_offset_.clear();
    per_client_events_by_ext_correlation_.clear();
    tracer_correlation_to_unique_correlation_.clear();
    events_pending_client_mapping_.clear();
}

Events* GPUTracerManager::GetEventListForUniqueCorrelationId(uint64_t unique_correlation_id) {
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

void GPUTracerManager::MapEventToClient(uint64_t tracer_correlation_id, EventRecord&& event) {
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

void GPUTracerManager::MapEventsToClient(uint64_t unique_correlation_id, std::vector<EventRecord>&& events) {
    auto p_event_list = GetEventListForUniqueCorrelationId(unique_correlation_id);
    if (p_event_list != nullptr) {
      p_event_list->insert(p_event_list->end(),
                           std::make_move_iterator(events.begin()),
                           std::make_move_iterator(events.end()));
    }
}

void GPUTracerManager::DeferEventMapping(EventRecord&& event, uint64_t tracer_correlation_id) {
    events_pending_client_mapping_[tracer_correlation_id].emplace_back(std::move(event));
}

void GPUTracerManager::NotifyNewCorrelation(uint64_t tracer_correlation_id, uint64_t unique_correlation_id) {
    tracer_correlation_to_unique_correlation_[tracer_correlation_id] = unique_correlation_id;
    auto pending_it = events_pending_client_mapping_.find(tracer_correlation_id);
    if (pending_it == events_pending_client_mapping_.end()) {
      return;
    }
    // Map the pending events to the right client
    MapEventsToClient(tracer_correlation_id, std::move(pending_it->second));
    events_pending_client_mapping_.erase(pending_it);
}


// Implementation of GPUProfileBase
void GPUProfilerBase::MergeEvents(std::map<uint64_t, Events>& events_to_merge, Events& events) {
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

          // roctracer doesn't use Jan 1 1970 as an epoch for its timestamps.
          // So, we adjust the timestamp here to something sensible.
          evt.ts = event_iter->ts + increment;
          ++increment;
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

} /* end namespace profiling */
} /* end namespace onnxruntime */
