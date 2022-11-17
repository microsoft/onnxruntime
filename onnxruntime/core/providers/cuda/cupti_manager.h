#pragma once

#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING)

#include <atomic>
#include <mutex>
#include <vector>

#include <cupti.h>

#include "core/common/gpu_profiler_common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace profiling {

class CUPTIManager : public GPUTracerManager
{
public:
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CUPTIManager);
    ~CUPTIManager();
    static CUPTIManager& GetInstance();

protected:
  bool PushUniqueCorrelation(uint64_t unique_cid) override;
  void PopUniqueCorrelation(uint64_t& popped_unique_cid) override;
  bool OnStartLogging() override;
  void OnStopLogging() override;
  void ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                              const TimePoint& start_time) override;
  void FlushActivities() override;

private:
    static constexpr size_t kActivityBufferSize = 32 * 1024;
    static constexpr size_t kActivityBufferAlignSize = 8;

    // TODO: Is this even needed? malloc() is required to return
    // a memory block that meets the alignment requirements for _any_ data type.
    // On any platform that supports an 8-byte datatype (double? long long?)
    // this means that malloc() already returns memory aligned at
    // _at least_ 8 byte boundaries, rendering this additional alignment
    // redundant?
    static constexpr uint8_t* AlignBuffer(uint8_t* buffer, int align) {
        return (((uintptr_t)(buffer) & ((align)-1))
                ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))
                : (buffer));
    }

    CUPTIManager() = default;

    static void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
    static void CUPTIAPI BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size);
}; /* class CUPTIManager*/

#endif /* #if defined (USE_CUDA) && defined(ENABLE_CUDA_PROFILING) */

} /* namespace profiling */
} /* namespace onnxruntime */
