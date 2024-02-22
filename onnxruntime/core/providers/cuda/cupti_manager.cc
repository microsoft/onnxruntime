// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cupti_manager.h"

#include <memory>

namespace onnxruntime {
namespace profiling {

#if defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING)

static inline std::string GetMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "MemcpyHostToDevice";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "MemcpyDeviceToHost";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "MemcpyHostToDeviceArray";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "MemcpyDeviceArrayToHost";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "MemcpyDeviceArrayToDeviceArray";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "MemcpyDeviceArrayToDevice";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "MemcpyDeviceToDeviceArray";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "MemcpyDeviceToDevice";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "MemcpyHostToHost";
    default:
      break;
  }
  return "<unknown>";
}

CUPTIManager& CUPTIManager::GetInstance() {
  static CUPTIManager instance;
  return instance;
}

CUPTIManager::~CUPTIManager() {}

bool CUPTIManager::OnStartLogging() {
  if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME) == CUPTI_SUCCESS &&
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER) == CUPTI_SUCCESS &&
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == CUPTI_SUCCESS &&
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY) == CUPTI_SUCCESS &&
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) == CUPTI_SUCCESS &&
      cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted) == CUPTI_SUCCESS) {
    return true;
  } else {
    OnStopLogging();
    return false;
  }
}

void CUPTIManager::OnStopLogging() {
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
}

bool CUPTIManager::PushUniqueCorrelation(uint64_t unique_cid) {
  auto res = cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, unique_cid);
  return res == CUPTI_SUCCESS;
}

void CUPTIManager::PopUniqueCorrelation(uint64_t& popped_unique_cid) {
  auto res = cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &popped_unique_cid);
  if (res != CUPTI_SUCCESS) {
    popped_unique_cid = 0;
  }
}

void CUPTIManager::FlushActivities() {
  cuptiActivityFlushAll(1);
}

uint64_t CUPTIManager::GetGPUTimestampInNanoseconds() {
  uint64_t result;
  if (cuptiGetTimestamp(&result) != CUPTI_SUCCESS) {
    ORT_THROW("Could not retrieve timestamp from GPU!");
  }
  return result;
}

void CUPTIManager::ProcessActivityBuffers(const std::vector<ProfilerActivityBuffer>& buffers,
                                          const TimePoint& start_time) {
  auto start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();
  for (auto const& buffer : buffers) {
    auto size = buffer.GetSize();
    if (size == 0) {
      continue;
    }
    CUpti_Activity* record = nullptr;
    CUptiResult status;
    do {
      EventRecord event;
      status = cuptiActivityGetNextRecord(reinterpret_cast<uint8_t*>(const_cast<char*>(buffer.GetData())), size, &record);
      if (status == CUPTI_SUCCESS) {
        if (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL == record->kind ||
            CUPTI_ACTIVITY_KIND_KERNEL == record->kind) {
          CUpti_ActivityKernel3* kernel = (CUpti_ActivityKernel3*)record;
          std::unordered_map<std::string, std::string> args{
              {"stream", std::to_string(kernel->streamId)},
              {"grid_x", std::to_string(kernel->gridX)},
              {"grid_y", std::to_string(kernel->gridY)},
              {"grid_z", std::to_string(kernel->gridZ)},
              {"block_x", std::to_string(kernel->blockX)},
              {"block_y", std::to_string(kernel->blockY)},
              {"block_z", std::to_string(kernel->blockZ)},
          };

          std::string name{demangle(kernel->name)};

          new (&event) EventRecord{
              /* cat = */ EventCategory::KERNEL_EVENT,
              /* pid = */ -1,
              /* tid = */ -1,
              /* name = */ std::move(name),
              /* ts = */ (int64_t)(kernel->start - start_time_ns) / 1000,
              /* dur = */ (int64_t)(kernel->end - kernel->start) / 1000,
              /* args = */ std::move(args)};
          MapEventToClient(kernel->correlationId, std::move(event));
        } else if (CUPTI_ACTIVITY_KIND_MEMCPY == record->kind) {
          CUpti_ActivityMemcpy* mmcpy = (CUpti_ActivityMemcpy*)record;
          std::string name{GetMemcpyKindString((CUpti_ActivityMemcpyKind)mmcpy->copyKind)};
          std::unordered_map<std::string, std::string> args{
              {"stream", std::to_string(mmcpy->streamId)},
              {"grid_x", "-1"},
              {"grid_y", "-1"},
              {"grid_z", "-1"},
              {"block_x", "-1"},
              {"block_y", "-1"},
              {"block_z", "-1"},
          };
          new (&event) EventRecord{
              /* cat = */ EventCategory::KERNEL_EVENT,
              /* pid = */ -1,
              /* tid = */ -1,
              /* name = */ std::move(name),
              /* ts = */ (int64_t)(this->NormalizeGPUTimestampToCPUEpoch(mmcpy->start) - start_time_ns) / 1000,
              /* dur = */ (int64_t)(mmcpy->end - mmcpy->start) / 1000,
              /* args = */ std::move(args)};
          MapEventToClient(mmcpy->correlationId, std::move(event));
        } else if (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION == record->kind) {
          auto correlation = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record);
          NotifyNewCorrelation(correlation->correlationId, correlation->externalId);
        }
      }
    } while (status == CUPTI_SUCCESS);
  } /* for */
}

void CUPTIAPI CUPTIManager::BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  // ProfilerActivityBuffer expects a char[], match up new[] and delete[] types just to be safe!
  // Note on ownership: This method is a callback that is invoked whenever CUPTI needs
  // a new buffer to record trace events. We allocate the buffer here, and eventually
  // CUPTI returns the buffer to us via the BufferCompleted callback.
  // In the BufferCompleted callback, we pass the returned buffer into a ProfilerActivityBuffer
  // object, which then assumes ownership of the buffer. RAII semantics then delete/free the
  // buffer whenever the ProfilerActivityBuffer is destroyed.
  auto buf = new char[kActivityBufferSize];
  *size = kActivityBufferSize;
  *maxNumRecords = 0;
  *buffer = reinterpret_cast<uint8_t*>(buf);
}

void CUPTIAPI CUPTIManager::BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size) {
  auto& instance = GetInstance();
  std::unique_ptr<char[]> buffer_ptr;
  buffer_ptr.reset(reinterpret_cast<char*>(buffer));
  instance.EnqueueActivityBuffer(
      ProfilerActivityBuffer::CreateFromPreallocatedBuffer(std::move(buffer_ptr), valid_size));
}

#endif /* defined(USE_CUDA) && defined(ENABLE_CUDA_PROFILING) */

}  // namespace profiling
}  // namespace onnxruntime
