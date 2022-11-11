#include "cupti_manager.h"

namespace onnxruntime {
namespace profiling {

CUPTIManager& CUPTIManager::GetInstance() {
  static CUPTIManager instance;
  return instance;
}

CUPTIManager::~CUPTIManager() {
    StopLogging();
    Clear();
}

uint64_t CUPTIManager::RegisterClient() {
  std::lock_guard<std::mutex> lock(cupti_manager_mutex_);
  auto res = next_client_id_++;
  per_client_events_by_ext_correlation_.insert({res, {}});
  ++num_active_clients_;
  return res;
}

void CUPTIManager::DeregisterClient(uint64_t client_handle) {
    std::lock_guard<std::mutex> lock(cupti_manager_mutex_);
    per_client_events_by_ext_correlation_.erase(client_handle);
    --num_active_clients_;
    if (num_active_clients_ == 0) {
        StopLogging();
    }
}

void CUPTIManager::StartLogging() {
    std::lock_guard<std::mutex> lock(cupti_manager_mutex_);
    if (logging_enabled_)  {
        return;
    }
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) == CUPTI_SUCCESS &&
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted) == CUPTI_SUCCESS) {
      logging_enabled_ = true;
    } else {
      StopLogging();
      logging_enabled_ = false;
    }
}

void CUPTIManager::StopLogging() {
  std::lock_guard<std::mutex> lock(cupti_manager_mutex_);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
  logging_enabled_ = false;
}

void CUPTIManager::Clear() {
    unprocessed_activity_buffers_.clear();
    unique_correlation_id_to_client_offset_.clear();
    per_client_events_by_ext_correlation_.clear();
    cupti_correlation_to_unique_correlation_.clear();
}

bool CUPTIManager::PushCorrelation(uint64_t client_handle,
                                  uint64_t external_correlation_id,
                                  TimePoint origin) {
    std::lock_guard<std::mutex> lock(cupti_manager_mutex_);
    if (!logging_enabled_) {
        return false;
    }
    if (per_client_events_by_ext_correlation_.find(client_handle) ==
        per_client_events_by_ext_correlation_.end()) {
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

    uint64_t offset =
      std::chrono::duration_cast<std::chrono::microseconds>(profiling_start_time.time_since_epoch()).count();
    auto unique_cid = external_correlation_id + offset;

    cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, unique_cid);

    unique_correlation_id_to_client_offset_[unique_cid] = std::make_pair(client_handle, offset);
    return true;
}

void CUPTIManager::PopCorrelation(uint64_t& popped_correlation_id) {
    popped_correlation_id = 0;
    std::lock_guard<std::mutex> lock(cupti_manager_mutex_);
    if (!logging_enabled_) {
        return;
    }

    uint64_t unique_cid;
    cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &unique_cid);
    // lookup the offset and subtract it before returning popped_external_correlation_id to the client
    auto client_it = unique_correlation_id_to_client_offset_.find(unique_cid);
    if (client_it == unique_correlation_id_to_client_offset_.end()) {
        return;
    }
    popped_correlation_id = unique_cid - client_it->second.second;
}

void CUPTIAPI CUPTIManager::BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    uint8_t* bfr = (uint8_t*)malloc(kActivityBufferSize + kActivityBufferAlignSize);
    *size = kActivityBufferSize;
    *buffer = AlignBuffer(bfr, kActivityBufferAlignSize);
    *maxNumRecords = 0;
}

void CUPTIAPI CUPTIManager::BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size) {
    auto instance = GetInstance();
    std::lock_guard<std::mutex> lock(instance.unprocessed_activity_buffers_lock_);
    instance.unprocessed_activity_buffers_.emplace_back(
        CUPTIActivityBuffer::CreateFromPreallocatedBuffer(reinterpret_cast<char*>(buffer), valid_size);
    );
}

void CUPTIManager::Consume(uint64_t client_handle, const TimePoint& start_time, std::map<uint64_t, Events>& events) {

}

} // namespace profiling
} // namespace onnxruntime
