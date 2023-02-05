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

#include "core/common/gpu_profiler_common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace profiling {

struct ApiCallRecord {
  uint32_t domain_;
  uint32_t cid_;
  hip_api_data_t api_data_{};
};

class RoctracerManager : public GPUTracerManager<RoctracerManager> {
  friend class GPUTracerManager<RoctracerManager>;

 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoctracerManager);
  ~RoctracerManager();
  static RoctracerManager& GetInstance();

 protected:
  bool PushUniqueCorrelation(uint64_t unique_cid);
  void PopUniqueCorrelation(uint64_t& popped_unique_cid);
  void OnStopLogging();
  bool OnStartLogging();
  void ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                              const TimePoint& start_time);
  void FlushActivities();
  uint64_t GetGPUTimestampInNanoseconds();

 private:
  RoctracerManager() = default;
  static void ActivityCallback(const char* begin, const char* end, void* arg);
  static void ApiCallback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
  bool CreateEventForActivityRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                    const ApiCallRecord& call_record, EventRecord& event);

  // Some useful constants for processing activity buffers
  static constexpr uint32_t HipOpMarker = 4606;

  std::mutex api_call_args_mutex_;
  InlinedHashMap<uint64_t, ApiCallRecord> api_call_args_;

  // The api calls to track
  static const std::vector<std::string> hip_api_calls_to_trace;
}; /* class RoctracerManager */

} /* end namespace profiling */
} /* end namespace onnxruntime*/
