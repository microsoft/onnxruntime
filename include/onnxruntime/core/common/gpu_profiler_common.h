#pragma once

#include "core/common/profiler_common.h"
#include "core/common/inlined_containers.h"

#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <utility>


namespace onnxruntime {
namespace profiling {

class ProfilerActivityBuffer {
 public:
  ProfilerActivityBuffer();
  ProfilerActivityBuffer(const char* data, size_t size);
  ProfilerActivityBuffer(const ProfilerActivityBuffer& other);
  ProfilerActivityBuffer(ProfilerActivityBuffer&& other);
  ProfilerActivityBuffer& operator=(const ProfilerActivityBuffer& other);
  ProfilerActivityBuffer& operator=(ProfilerActivityBuffer&& other);

  // accessors
  char* GetData() { return data_.get(); }
  const char* GetData() const { return data_.get(); }
  size_t GetSize() const { return size_; }

  static ProfilerActivityBuffer CreateFromPreallocatedBuffer(char* data, size_t size);

 private:
  std::unique_ptr<char[]> data_;
  size_t size_;
}; /* end class ProfilerActivityBuffer */

class GPUTracerManager
{
public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GPUTracerManager);
  virtual ~GPUTracerManager() {}

  uint64_t RegisterClient();
  void DeregisterClient(uint64_t client_handle);

  void StartLogging();
  void Consume(uint64_t client_handle, const TimePoint& start_time, std::map<uint64_t, Events>& events);
  bool PushCorrelation(uint64_t client_handle,
                       uint64_t external_correlation_id,
                       TimePoint profiling_start_time);
  void PopCorrelation(uint64_t& popped_external_correlation_id);
  void PopCorrelation();

protected:
  GPUTracerManager() = default;

  // Functional API to be implemented by subclasses
  virtual bool OnStartLogging() = 0;
  virtual void OnStopLogging() = 0;
  virtual void ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                                      const TimePoint& start_time) = 0;
  virtual bool PushUniqueCorrelation(uint64_t unique_cid) = 0;
  virtual void PopUniqueCorrelation(uint64_t& popped_unique_cid) = 0;
  virtual void FlushActivities() = 0;

  // Service API for subclasses
  void EnqueueActivityBuffer(ProfilerActivityBuffer&& buffer);
  // To be called by subclasses only from ProcessActivityBuffers
  void MapEventToClient(uint64_t tracer_correlation_id, EventRecord&& event);
  // To be called by subclasses only from ProcessActivityBuffers
  void NotifyNewCorrelation(uint64_t tracer_correlation_id, uint64_t unique_correlation_id);

private:
  void StopLogging();
  void Clear();
  Events* GetEventListForUniqueCorrelationId(uint64_t unique_correlation_id);
  void MapEventsToClient(uint64_t unique_correlation_id, std::vector<EventRecord>&& events);
  void DeferEventMapping(EventRecord&& event, uint64_t tracer_correlation_id);

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
  void MergeEvents(std::map<uint64_t, Events>& events_to_merge, Events& events);
}; /* class GPUProfilerBase */

// Convert a pointer to a hex string
static inline std::string PointerToHexString(const void* ptr) {
  std::ostringstream sstr;
  sstr << std::hex << ptr;
  return sstr.str();
}

} /* end namespace profiling */
} /* end namespace onnxruntime */
