#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hcc.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_roctx.h>

#include "core/common/profiler_common.h"


namespace onnxruntime{
namespace profiling {

class RoctracerActivityBuffer {
public:
  RoctracerActivityBuffer()
    : data_(nullptr), size_(0) {}

  RoctracerActivityBuffer(const uint8_t* data, size_t size)
    : data_((uint8_t*)malloc(size)), size_(size) {
    memcpy(data_, data, size);
  }

  RoctracerActivityBuffer(const RoctracerActivityBuffer& other)
    : RoctracerActivityBuffer(other.data_, other.size_) {}

  RoctracerActivityBuffer(RoctracerActivityBuffer&& other)
    : RoctracerActivityBuffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  RoctracerActivityBuffer& operator = (const RoctracerActivityBuffer& other);
  RoctracerActivityBuffer& operator = (RoctracerActivityBuffer&& other);

  ~RoctracerActivityBuffer();

  // accessors
  uint8_t* GetData() { return data_; }
  const uint8_t* GetData() const { return data_; }
  size_t GetSize() const { return size_; }

private:
  uint8_t* data_;
  size_t size_;
};

struct ApiCallRecord {
  uint32_t domain_;
  uint32_t cid_;
  hip_api_data_t api_data_ {};
};

class RoctracerManager
{
public:
  RoctracerManager(const RoctracerManager&) = delete;
  RoctracerManager& operator = (const RoctracerManager&) = delete;
  RoctracerManager() = default;
  ~RoctracerManager();

  static RoctracerManager& GetInstance();

  uint64_t RegisterClient();
  void DeregisterClient(uint64_t client_handle);

  void StartLogging();
  void StopLogging();
  void Consume(uint64_t client_handle, const TimePoint& start_time, std::map<uint64_t, Events>& events);

  bool PushCorrelation(uint64_t client_handle, uint64_t external_correlation_id);
  void PopCorrelation(uint64_t& popped_correlation_id);
  bool PopCorrelation();

private:
  static void ActivityCallback(const char* begin, const char* end, void* arg);
  static void ApiCallback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
  void ProcessActivityBuffers(const std::vector<RoctracerActivityBuffer>& buffers, const TimePoint& start_time);

  // Per-API (roughly) helpers for event construction.
  void CreateEventForKernelRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                  const ApiCallRecord& call_record, EventRecord& event);
  void CreateEventForMemsetRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                  const ApiCallRecord& call_record, EventRecord& event);
  void CreateEventForMemcpyRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                  const ApiCallRecord& call_record, EventRecord& event);
  void CreateEventForMemcpy2DRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                    const ApiCallRecord& call_record, EventRecord& event);
  void MapEventToClient(uint64_t external_correlation_id, EventRecord&& event);
  void MapEventsToClient(uint64_t external_correlation_id, Events&& events);

  // Some useful constants for processing activity buffers
  static constexpr uint32_t HipOpMarker = 4606;

  std::mutex unprocessed_activity_buffers_lock_;
  std::vector<RoctracerActivityBuffer> unprocessed_activity_buffers_;
  std::mutex activity_buffer_processor_mutex_;
  std::mutex api_call_args_lock_;
  std::unordered_map<uint64_t, ApiCallRecord> api_call_args_;

  // Keyed on external_correlation_id -> client_id/client_handle
  std::unordered_map<uint64_t, uint64_t> external_correlation_id_to_client_;

  // Keyed on roctracer_correlation_id -> external_correlation_id
  std::unordered_map<uint64_t, uint64_t> roctracer_correlation_to_external_correlation_;

  // client_id/client_handle -> external_correlation_id -> events
  std::mutex event_list_mutex_;
  std::unordered_map<uint64_t, std::map<uint64_t, Events>> per_client_events_by_ext_correlation_;
  uint64_t next_client_id_ = 1;
  bool logging_enabled_ = false;
  std::mutex roctracer_manager_mutex_;
  roctracer_pool_t* activity_pool_;

  // The api calls to track
  static const std::vector<std::string> hip_api_calls_to_trace;
};

} /* end namespace profiling */
} /* end namespace onnxruntime */


/*
}
}

class RoctracerActivityBuffer {
public:
  // data must be allocated using malloc.
  // Ownership is transferred to this object.
  RoctracerActivityBuffer(uint8_t* data, size_t validSize)
      : data_(data), validSize_(validSize) {}

  ~RoctracerActivityBuffer() {
    free(data_);
  }

  // Allocated by malloc
  uint8_t* data_{nullptr};

  // Number of bytes used
  size_t validSize_;
};

struct roctracerRow {
  roctracerRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end)
    : id(id), domain(domain), cid(cid), pid(pid), tid(tid), begin(begin), end(end) {}
  uint64_t id;  // correlation_id
  uint32_t domain;
  uint32_t cid;
  uint32_t pid;
  uint32_t tid;
  uint64_t begin;
  uint64_t end;
};

struct kernelRow : public roctracerRow {
  kernelRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
          , uint32_t tid, uint64_t begin, uint64_t end
          , const void *faddr, hipFunction_t function
          , unsigned int gx, unsigned int gy, unsigned int gz
          , unsigned int wx, unsigned int wy, unsigned int wz
          , size_t gss, hipStream_t stream)
    : roctracerRow(id, domain, cid, pid, tid, begin, end), functionAddr(faddr)
    , function(function), gridX(gx), gridY(gy), gridZ(gz)
    , workgroupX(wx), workgroupY(wy), workgroupZ(wz), groupSegmentSize(gss)
    , stream(stream) {}
  const void* functionAddr;
  hipFunction_t function;
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int workgroupX;
  unsigned int workgroupY;
  unsigned int workgroupZ;
  size_t groupSegmentSize;
  hipStream_t stream;
};

struct copyRow : public roctracerRow {
  copyRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end
             , const void* src, const void *dst, size_t size, hipMemcpyKind kind
             , hipStream_t stream)
    : roctracerRow(id, domain, cid, pid, tid, begin, end)
    , src(src), dst(dst), size(size), kind(kind), stream(stream) {}
  const void *src;
  const void *dst;
  size_t size;
  hipMemcpyKind kind;
  hipStream_t stream;
};

struct mallocRow : public roctracerRow {
  mallocRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end
             , const void* ptr, size_t size)
    : roctracerRow(id, domain, cid, pid, tid, begin, end)
    , ptr(ptr), size(size) {}
  const void *ptr;
  size_t size;
};


class RoctracerLogger {
 public:
  enum CorrelationDomain {
    begin,
    Default = begin,
    Domain0 = begin,
    Domain1,
    end,
    size = end
  };

  RoctracerLogger();
  RoctracerLogger(const RoctracerLogger&) = delete;
  RoctracerLogger& operator=(const RoctracerLogger&) = delete;

  virtual ~RoctracerLogger();

  static RoctracerLogger& singleton();

  static void pushCorrelationID(uint64_t id, CorrelationDomain type);
  static void popCorrelationID(CorrelationDomain type);

  void startLogging();
  void stopLogging();
  void clearLogs();

 private:
  bool registered_{false};
  void endTracing();

  roctracer_pool_t *hccPool_{NULL};
  static void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
  static void activity_callback(const char* begin, const char* end, void* arg);

  ApiIdList loggedIds_;

  // Api callback data
  std::deque<roctracerRow> rows_;
  std::deque<kernelRow> kernelRows_;
  std::deque<copyRow> copyRows_;
  std::deque<mallocRow> mallocRows_;
  std::map<uint64_t,uint64_t> externalCorrelations_[CorrelationDomain::size];	// tracer -> ext

  std::unique_ptr<std::list<RoctracerActivityBuffer>> gpuTraceBuffers_;
  bool externalCorrelationEnabled_{true};

  friend class onnxruntime::profiling::RocmProfiler;
};
*/
