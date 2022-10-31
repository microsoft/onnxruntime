// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

#include <chrono>
#include <time.h>

#include "core/common/profiler_common.h"
#include "core/providers/rocm/rocm_profiler.h"
#include "core/providers/rocm/roctracer_manager.h"

#define BSIZE 4096

typedef uint64_t timestamp_t;
static timestamp_t timespec_to_ns(const timespec& time) {
  return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
}

namespace onnxruntime {
namespace profiling {

RocmProfiler::RocmProfiler() {
  auto& manager = RoctracerManager::GetInstance();
  client_handle_ = manager.RegisterClient();
}

RocmProfiler::~RocmProfiler() {
  auto& manager = RoctracerManager::GetInstance();
  manager.DeregisterClient(client_handle_);
}

bool RocmProfiler::StartProfiling()
{
  auto& manager = RoctracerManager::GetInstance();
  manager.StartLogging();
}

void RocmProfiler::EndProfiling(TimePoint start_time, Events& events)
{
  auto& manager = RoctracerManager::GetInstance();
  std::map<uint64_t, Events> event_map;
  manager.Consume(client_handle_, start_time, event_map);

  Events merged_events;

  auto event_iter = std::make_move_iterator(events.begin());
  auto event_end = std::make_move_iterator(events.end());
  for (auto& map_iter : event_map) {
    auto ts = static_cast<long long>(map_iter.first);
    while (event_iter != event_end && event_iter->ts < ts) {
      merged_events.emplace_back(*event_iter);
      ++event_iter;
    }

    if (event_iter != event_end && event_iter->ts == ts) {
      for (auto& evt : map_iter.second) {
        evt.args["op_name"] = event_iter->args["op_name"];
      }
      merged_events.emplace_back(*event_iter);
      ++event_iter;
    }

    merged_events.insert(merged_events.end(),
                         std::make_move_iterator(map_iter.second.begin()),
                         std::make_move_iterator(map_iter.second.end()));
  }

  // move any remaining events
  merged_events.insert(merged_events.end(), event_iter, event_end);
  std::swap(events, merged_events);

  // // Generate EventRecords
  // int64_t profiling_start = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();

  // // Wrong clock again - all the cool kids are doing it
  // timespec t0, t1, t00;
  // clock_gettime(CLOCK_REALTIME, &t0);
  // clock_gettime(CLOCK_MONOTONIC, &t1);
  // clock_gettime(CLOCK_REALTIME, &t00);
  // const uint64_t toffset = (timespec_to_ns(t0) >> 1) + (timespec_to_ns(t00) >> 1) - timespec_to_ns(t1);
  // profiling_start = profiling_start - toffset;

  // char buff[BSIZE];

  // for (auto &item : d->rows_) {
  //   std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""}};
  //   addEventRecord(item, profiling_start, args, event_map);
  // }

  // for (auto &item : d->mallocRows_) {
  //   snprintf(buff, BSIZE, "%p", item.ptr);
  //   const std::string arg_ptr{buff};

  //   std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""},
  //                                           {"ptr", arg_ptr},
  //                                           {"size", std::to_string(item.size)}
  //                                           };
  //   addEventRecord(item, profiling_start, args, event_map);
  // }

  // for (auto &item : d->copyRows_) {
  //   snprintf(buff, BSIZE, "%p", item.stream);
  //   const std::string arg_stream{buff};
  //   snprintf(buff, BSIZE, "%p", item.src);
  //   const std::string arg_src{buff};
  //   snprintf(buff, BSIZE, "%p", item.dst);
  //   const std::string arg_dst{buff};

  //   std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""},
  //                                           {"stream", arg_stream},
  //                                           {"src", arg_src},
  //                                           {"dst", arg_dst},
  //                                           {"size", std::to_string(item.size)},
  //                                           {"kind", std::to_string(item.kind)},
  //                                           };
  //   addEventRecord(item, profiling_start, args, event_map);
  //   copyLaunches[item.id] = &item;
  // }

  // for (auto &item : d->kernelRows_) {
  //   snprintf(buff, BSIZE, "%p", item.stream);
  //   const std::string arg_stream{buff};
  //   if (item.functionAddr)
  //       snprintf(buff, BSIZE, "%s", demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream)).c_str());
  //   else if (item.function)
  //       snprintf(buff, BSIZE, "%s", demangle(hipKernelNameRef(item.function)).c_str());
  //   else
  //       snprintf(buff, BSIZE, " ");
  //   const std::string arg_kernel{buff};

  //   std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""},
  //                                           {"stream", arg_stream},
  //                                           {"kernel", arg_kernel},
  //                                           {"grid_x", std::to_string(item.gridX)},
  //                                           {"grid_y", std::to_string(item.gridY)},
  //                                           {"grid_z", std::to_string(item.gridZ)},
  //                                           {"block_x", std::to_string(item.workgroupX)},
  //                                           {"block_y", std::to_string(item.workgroupY)},
  //                                           {"block_z", std::to_string(item.workgroupZ)},
  //                                           };
  //   addEventRecord(item, profiling_start, args, event_map);
  //   kernelLaunches[item.id] = &item;
  // }

  // // Async Ops - e.g. "Kernel"

  // for (auto& buffer : *d->gpuTraceBuffers_) {
  //   const roctracer_record_t* record = (const roctracer_record_t*)(buffer.data_);
  //   const roctracer_record_t* end_record = (const roctracer_record_t*)(buffer.data_ + buffer.validSize_);
  //   while (record < end_record) {
  //     std::unordered_map<std::string, std::string> args;
  //     std::string name = roctracer_op_string(record->domain, record->op, record->kind);

  //     // Add kernel args if we have them
  //     auto kit = kernelLaunches.find(record->correlation_id);
  //     if (kit != kernelLaunches.end()) {
  //       auto &item = *(*kit).second;
  //       snprintf(buff, BSIZE, "%p", item.stream);
  //       args["stream"] = std::string(buff);
  //       args["grid_x"] = std::to_string(item.gridX);
  //       args["grid_y"] = std::to_string(item.gridY);
  //       args["grid_z"] = std::to_string(item.gridZ);
  //       args["block_x"] = std::to_string(item.workgroupX);
  //       args["block_y"] = std::to_string(item.workgroupY);
  //       args["block_z"] = std::to_string(item.workgroupZ);
  //       if (item.functionAddr != nullptr) {
  //         name = demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream)).c_str();
  //       }
  //       else if (item.function != nullptr) {
  //         name = demangle(hipKernelNameRef(item.function)).c_str();
  //       }
  //     }

  //     // Add copy args if we have them
  //     auto cit = copyLaunches.find(record->correlation_id);
  //     if (cit != copyLaunches.end()) {
  //       auto &item = *(*cit).second;
  //       snprintf(buff, BSIZE, "%p", item.stream);
  //       args["stream"] = std::string(buff);
  //       snprintf(buff, BSIZE, "%p", item.src);
  //       args["dst"] = std::string(buff);
  //       snprintf(buff, BSIZE, "%p", item.dst);
  //       args["src"] = std::string(buff);
  //       args["size"] = std::to_string(item.size);
  //       args["kind"] = std::to_string(item.kind);
  //     }

  //     EventRecord event{
  //         onnxruntime::profiling::KERNEL_EVENT,
  //         static_cast<int>(record->device_id),
  //         static_cast<int>(record->queue_id),
  //         name,
  //         static_cast<int64_t>((record->begin_ns - profiling_start) / 1000),
  //         static_cast<int64_t>((record->end_ns - record->begin_ns) / 1000),
  //         std::move(args)};

  //      // FIXME: deal with missing ext correlation
  //      auto extId = d->externalCorrelations_[RoctracerLogger::CorrelationDomain::Default][record->correlation_id];
  //      if (event_map.find(extId) == event_map.end()) {
  //        event_map.insert({extId, {event}});
  //      }
  //      else {
  //        event_map[extId].push_back(std::move(event));
  //      }

  //      roctracer_next_record(record, &record);
  //   }
  // }

  // // General
  // auto insert_iter = events.begin();
  // for (auto& map_iter : event_map) {
  //   auto ts = static_cast<long long>(map_iter.first);
  //   while (insert_iter != events.end() && insert_iter->ts < ts) {
  //     insert_iter++;
  //   }
  //   if (insert_iter != events.end() && insert_iter->ts == ts) {
  //     for (auto& evt_iter : map_iter.second) {
  //       evt_iter.args["op_name"] = insert_iter->args["op_name"];
  //     }
  //     insert_iter = events.insert(insert_iter+1, map_iter.second.begin(), map_iter.second.end());
  //   } else {
  //     insert_iter = events.insert(insert_iter, map_iter.second.begin(), map_iter.second.end());
  //   }
  //   while (insert_iter != events.end() && insert_iter->cat == EventCategory::KERNEL_EVENT) {
  //     insert_iter++;
  //   }
  // }
}

void RocmProfiler::Start(uint64_t id)
{
  auto& manager = RoctracerManager::GetInstance();
  manager.PushCorrelation(client_handle_, id);
}

void RocmProfiler::Stop(uint64_t id)
{
  auto& manager = RoctracerManager::GetInstance();
  uint64_t unused;
  manager.PopCorrelation(unused);
}

}  // namespace profiling
}  // namespace onnxruntime
#endif
