#pragma once


#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING)

#include <atomic>
#include <mutex>
#include <vector>

#include <cupti.h>

#include "core/platform/ort_mutex.h"
#include "core/common/profiler_common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace profiling {

using CUPTIActivityBuffer = ProfilerActivityBuffer;

class CUPTIManager final
{
public:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CUPTIManager);
    ~CUPTIManager();
    static CUPTIManager& GetInstance();
    uint64_t RegisterClient();
    void DeregisterClient(uint64_t client_handle);

    void StartLogging();
    void Consume(uint64_t client_handle, const TimePoint& start_time, std::map<uint64_t, Events>& events);
    bool PushCorrelation(uint64_t client_handle, uint64_t external_correlation_id, TimePoint profiling_start_time);
    void PopCorrelation(uint64_t& popped_correlation_id);

private:
    static constexpr size_t kActivityBufferSize = 32 * 1024;
    static constexpr size_t kActivityBufferAlignSize = 8;
    static constexpr void* AlignBuffer(buffer, align) {
        return (((uintptr_t)(buffer) & ((align)-1))
                ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))
                : (buffer));
    }

    CUPTIManager() = default;
    static void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
    static void CUPTIAPI BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size);
    void StopLogging();
    void Clear();

    std::mutex unprocessed_activity_buffers_lock_;
    std::vector<CUPTIActivityBuffer> unprocessed_activity_buffers_;
    std::mutex activity_buffer_processor_mutex_;
    std::mutex callback_mutex_;
    std::mutex cupti_manager_mutex_;
    uint64_t next_client_id_ = 1;
    uint64_t num_active_clients_ = 0;
    bool logging_enabled_ = false;

    // Keyed on unique_correlation_id -> (client_id/client_handle, offset)
    // unique_correlation_id - offset == external_correlation_id
    InlinedHashMap<uint64_t, std::pair<uint64_t, uint64_t>> unique_correlation_id_to_client_offset_;

    // Keyed on cupti_correlation_id -> unique_correlation_id
    InlinedHashMap<uint64_t, uint64_t> cupti_correlation_to_unique_correlation_;

      // client_id/client_handle -> external_correlation_id -> events
    InlinedHashMap<uint64_t, std::map<uint64_t, Events>> per_client_events_by_ext_correlation_;
}; /* class CUPTIManager*/

#endif /* #if defined (USE_CUDA) && defined(ENABLE_CUDA_PROFILING) */

} /* namespace profiling */
} /* namespace onnxruntime */
