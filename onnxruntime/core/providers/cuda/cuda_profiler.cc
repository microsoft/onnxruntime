// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cuda_profiler.h"
#include <map>
#include <string>
#include <iostream>

namespace onnxruntime {

namespace profiling {

auto KEVENT = onnxruntime::profiling::KERNEL_EVENT;
onnxruntime::OrtMutex CudaProfiler::mtx;
std::atomic_flag CudaProfiler::enabled{0};
std::vector<CudaProfiler::KernelStat> CudaProfiler::stats;
std::unordered_map<uint32_t, uint64_t> CudaProfiler::id_map;
bool CudaProfiler::initialized{false};

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
                           kernel->correlationId});
        } else if (CUPTI_ACTIVITY_KIND_MEMCPY == record->kind) {
          CUpti_ActivityMemcpy4* mmcpy = (CUpti_ActivityMemcpy4*)record;
          stats.push_back({GetMemcpyKindString((CUpti_ActivityMemcpyKind)mmcpy->copyKind),
                           mmcpy->streamId, -1, -1, -1, -1, -1, -1,
                           static_cast<int64_t>(mmcpy->start),
                           static_cast<int64_t>(mmcpy->end),
                           mmcpy->correlationId});
        } else if (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION == record->kind) {
          auto correlation = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record);
          id_map.insert({correlation->correlationId, correlation->externalId});
          //std::cout << "cuda correlation event captured"  << std::endl;
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
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) == CUPTI_SUCCESS &&
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted) == CUPTI_SUCCESS) {
      initialized = true;
      return true;
    }
  }
  return false;
}

void CudaProfiler::EndProfiling(TimePoint start_time, Events& events) {
  std::map<uint64_t, std::vector<EventRecord>> event_map;
  if (enabled.test_and_set()) {
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    if (initialized) {
      cuptiActivityFlushAll(1);
      std::lock_guard<onnxruntime::OrtMutex> lock(mtx);
      int64_t profiling_start = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();
      for (const auto& stat : stats) {
        std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", stat.name_},
                                                                           {"stream", std::to_string(stat.stream_)},
                                                                           {"grid_x", std::to_string(stat.grid_x_)},
                                                                           {"grid_y", std::to_string(stat.grid_y_)},
                                                                           {"grid_z", std::to_string(stat.grid_z_)},
                                                                           {"block_x", std::to_string(stat.block_x_)},
                                                                           {"block_y", std::to_string(stat.block_y_)},
                                                                           {"block_z", std::to_string(stat.block_z_)}};
        EventRecord event{
            KEVENT, -1, -1, stat.name_, DUR(profiling_start, stat.stop_), DUR(stat.start_, stat.stop_), {args.begin(), args.end()}};
        auto ts = id_map[stat.correlation_id];
        if (event_map.find(ts) == event_map.end()) {
          event_map.insert({ts, {event}});
        } else {
          event_map[ts].push_back(std::move(event));
        }
      }
      stats.clear();
      cuptiFinalize();
    }
  }
  id_map.clear();
  enabled.clear();
  auto insert_iter = events.begin();
  for (auto& map_iter : event_map) {
    auto ts = static_cast<long long>(map_iter.first);
    while (insert_iter != events.end() && insert_iter->ts < ts) {
      insert_iter++;
    }
    if (insert_iter != events.end() && insert_iter != events.begin() && insert_iter->ts > ts) {
      insert_iter--;
    }
    insert_iter = events.insert(insert_iter, map_iter.second.begin(), map_iter.second.end());
    while (insert_iter != events.end() && insert_iter->cat == EventCategory::KERNEL_EVENT) {
      insert_iter++;
    }
  }
} 

void CudaProfiler::Start(uint64_t id) {
  cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, id);
  //std::cout << "cuda correlation id pushed" << std::endl;
}

void CudaProfiler::Stop(uint64_t) {
  uint64_t last_id{0};
  cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &last_id);
  //std::cout << "cuda correlation id popped" << std::endl;
}

}  // namespace profiling
}  // namespace onnxruntime