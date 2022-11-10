#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>

#include <hip/hip_runtime_api.h>
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hcc.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_roctx.h>

#include "core/common/profiler_common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace profiling {

class RoctracerActivityBuffer {
 public:
  RoctracerActivityBuffer()
      : data_(nullptr), size_(0) {}

  RoctracerActivityBuffer(const char* data, size_t size)
      : data_(std::make_unique<char[]>(size)), size_(size) {
    memcpy(data_.get(), data, size);
  }

  RoctracerActivityBuffer(const RoctracerActivityBuffer& other)
      : RoctracerActivityBuffer(other.data_.get(), other.size_) {}

  RoctracerActivityBuffer(RoctracerActivityBuffer&& other)
      : RoctracerActivityBuffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  RoctracerActivityBuffer& operator=(const RoctracerActivityBuffer& other);
  RoctracerActivityBuffer& operator=(RoctracerActivityBuffer&& other);

  // accessors
  char* GetData() { return data_.get(); }
  const char* GetData() const { return data_.get(); }
  size_t GetSize() const { return size_; }

 private:
  std::unique_ptr<char[]> data_;
  size_t size_;
};

struct ApiCallRecord {
  uint32_t domain_;
  uint32_t cid_;
  hip_api_data_t api_data_{};
};

class RoctracerManager {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoctracerManager);
  ~RoctracerManager();

  static RoctracerManager& GetInstance();

  uint64_t RegisterClient();
  void DeregisterClient(uint64_t client_handle);

  void StartLogging();
  void Consume(uint64_t client_handle, const TimePoint& start_time, std::map<uint64_t, Events>& events);

  bool PushCorrelation(uint64_t client_handle, uint64_t external_correlation_id, TimePoint profiling_start_time);
  void PopCorrelation(uint64_t& popped_correlation_id);
  bool PopCorrelation();

 private:
  RoctracerManager() = default;
  static void ActivityCallback(const char* begin, const char* end, void* arg);
  static void ApiCallback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
  void ProcessActivityBuffers(const std::vector<RoctracerActivityBuffer>& buffers,
                              const TimePoint& start_time);
  bool CreateEventForActivityRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                    const ApiCallRecord& call_record, EventRecord& event);
  Events* GetEventListForUniqueCorrelationId(uint64_t unique_correlation_id);
  void MapEventToClient(uint64_t external_correlation_id, EventRecord&& event);
  void MapEventsToClient(uint64_t external_correlation_id, Events&& events);
  void StopLogging();
  void Clear();

  // Some useful constants for processing activity buffers
  static constexpr uint32_t HipOpMarker = 4606;

  std::mutex unprocessed_activity_buffers_lock_;
  std::vector<RoctracerActivityBuffer> unprocessed_activity_buffers_;
  std::mutex activity_buffer_processor_mutex_;
  std::mutex api_call_args_lock_;
  InlinedHashMap<uint64_t, ApiCallRecord> api_call_args_;

  // Keyed on unique_correlation_id -> (client_id/client_handle, offset)
  // unique_correlation_id - offset == external_correlation_id
  InlinedHashMap<uint64_t, std::pair<uint64_t, uint64_t>> unique_correlation_id_to_client_offset_;

  // Keyed on roctracer_correlation_id -> unique_correlation_id
  InlinedHashMap<uint64_t, uint64_t> roctracer_correlation_to_unique_correlation_;

  // client_id/client_handle -> external_correlation_id -> events
  InlinedHashMap<uint64_t, std::map<uint64_t, Events>> per_client_events_by_ext_correlation_;
  uint64_t next_client_id_ = 1;
  uint64_t num_active_clients_ = 0;
  bool logging_enabled_ = false;
  std::mutex roctracer_manager_mutex_;

  // Keyed on roctracer correlation_id, keeps track of activity records
  // for which we haven't established the external_correlation_id yet.
  InlinedHashMap<uint64_t, std::vector<EventRecord>> events_pending_client_mapping_;

  // The api calls to track
  static const std::vector<std::string> hip_api_calls_to_trace;
};

} /* end namespace profiling */
} /* end namespace onnxruntime*/
