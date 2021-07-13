// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cuda_profiler.h"
#include <string>

namespace onnxruntime {

namespace profiling {

auto KEVENT = onnxruntime::profiling::KERNEL_EVENT;
onnxruntime::OrtMutex CudaProfiler::mtx;
std::atomic_flag CudaProfiler::enabled{0};
std::vector<CudaProfiler::KernelStat> CudaProfiler::stats;
bool CudaProfiler::initialized{false};
TimePoint CudaProfiler::start_time;

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))
#define DUR(s, e) std::lround(static_cast<double>(e - s) / 1000)

static const char* GetMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
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

void CUPTIAPI CudaProfiler::BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  uint8_t* bfr = (uint8_t*)malloc(BUF_SIZE + ALIGN_SIZE);
  //ORT_ENFORCE(bfr, "Failed to allocate memory for cuda kernel profiling.");
  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI CudaProfiler::BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  CUptiResult status;
  CUpti_Activity* record = NULL;
  if (validSize > 0) {
    std::lock_guard<onnxruntime::OrtMutex> lock(CudaProfiler::mtx);
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        if (CUPTI_ACTIVITY_KIND_KERNEL == record->kind) {
          CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*)record;
          stats.push_back({kernel->name, kernel->streamId,
                           kernel->gridX, kernel->gridY, kernel->gridZ,
                           kernel->blockX, kernel->blockY, kernel->blockZ,
                           static_cast<int64_t>(kernel->start),
                           static_cast<int64_t>(kernel->end),
                           static_cast<int64_t>(kernel->correlationId)});
        } else if (CUPTI_ACTIVITY_KIND_MEMCPY == record->kind) {
          CUpti_ActivityMemcpy4* mmcpy = (CUpti_ActivityMemcpy4*)record;
          stats.push_back({GetMemcpyKindString((CUpti_ActivityMemcpyKind)mmcpy->copyKind),
                           mmcpy->streamId, -1, -1, -1, -1, -1, -1,
                           static_cast<int64_t>(mmcpy->start),
                           static_cast<int64_t>(mmcpy->end),
                           static_cast<int64_t>(mmcpy->correlationId)});
        }
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
    } while (1);
  }
  free(buffer);
}

bool CudaProfiler::StartProfiling() {
  if (!enabled.test_and_set()) {
    start_time = std::chrono::high_resolution_clock::now();
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY) == CUPTI_SUCCESS &&
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted) == CUPTI_SUCCESS) {
      initialized = true;
      return true;
    }
  }
  return false;
}

Events CudaProfiler::StopProfiling() {
  Events events;
  if (enabled.test_and_set()) {
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    if (initialized) {
      cuptiActivityFlushAll(1);
      std::lock_guard<onnxruntime::OrtMutex> lock(mtx);
      int64_t profiling_start = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();
      for (const auto& stat : stats) {
        std::initializer_list<std::pair<std::string, std::string>> args = {{"stream", std::to_string(stat.stream_)},
                                                                           {"grid_x", std::to_string(stat.grid_x_)},
                                                                           {"grid_y", std::to_string(stat.grid_y_)},
                                                                           {"grid_z", std::to_string(stat.grid_z_)},
                                                                           {"block_x", std::to_string(stat.block_x_)},
                                                                           {"block_y", std::to_string(stat.block_y_)},
                                                                           {"block_z", std::to_string(stat.block_z_)},
                                                                           {"correlation_id", std::to_string(stat.correlation_id)}};
        events.push_back({KEVENT, -1, -1, stat.name_, DUR(profiling_start, stat.stop_), DUR(stat.start_, stat.stop_), {args.begin(), args.end()}});
      }
      stats.clear();
      cuptiFinalize();
    } else {
      std::initializer_list<std::pair<std::string, std::string>> args;
      events.push_back({KEVENT, -1, -1, "not_available_due_to_cupti_error", 0, 0, {args.begin(), args.end()}});
    }
  }
  enabled.clear();
  return events;
}

}  // namespace profiling
}  // namespace onnxruntime