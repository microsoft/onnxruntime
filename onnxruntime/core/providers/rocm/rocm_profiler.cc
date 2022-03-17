// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_ROCM) && defined(ENABLE_ROCM_PROFILING)

#include <chrono>
#include <time.h>
//#include <fmt/format.h>

#include "rocm_profiler.h"
#include "RoctracerLogger.h"

typedef uint64_t timestamp_t;
static timestamp_t timespec_to_ns(const timespec& time) {
  return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
}

namespace onnxruntime {

namespace profiling {

RocmProfiler::RocmProfiler()
: d(&RoctracerLogger::singleton())
{
}

RocmProfiler::~RocmProfiler()
{
}

bool RocmProfiler::StartProfiling() 
{
    d->clearLogs();
    d->startLogging();
    return true;
}

void RocmProfiler::EndProfiling(TimePoint start_time, Events& events)
{
  d->stopLogging();
  fprintf(stderr, "------------  kernels: %ld, copies: %ld\n", d->kernelRows_.size(), d->copyRows_.size());

  std::map<uint64_t, std::vector<EventRecord>> event_map;
  std::map<uint64_t, kernelRow*> kernelLaunches;   // correlation id -> kernel info
  std::map<uint64_t, copyRow*> copyLaunches;     // correlation id -> copy info
    
  // Generate EventRecords
  int64_t profiling_start = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();

  // Wrong clock again - all the cool kids are doing it
  timespec t0, t1, t00;
  clock_gettime(CLOCK_REALTIME, &t0);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  clock_gettime(CLOCK_REALTIME, &t00);
  const uint64_t toffset = (timespec_to_ns(t0) >> 1) + (timespec_to_ns(t00) >> 1) - timespec_to_ns(t1);
  profiling_start = profiling_start - toffset;

  // FIXME: disable api calls for now.  Need a a formatter.  Is this data desired?
#if 0
  fprintf(stderr, "base\n");
  for (auto &item : d->rows_) {
    std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""}};
    addEventRecord(item, profiling_start, args, event_map);
  }

  fprintf(stderr, "malloc\n");
  for (auto &item : d->mallocRows_) {
    std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""},
                                            //{"ptr", fmt::format("{}", item.ptr},
                                            {"size", std::to_string(item.size)}
                                            };
    addEventRecord(item, profiling_start, args, event_map);
  }

  fprintf(stderr, "copy\n");
  for (auto &item : d->copyRows_) {
    std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""},
                                            //{"stream", fmt::format("{}", item.stream}},
                                            //{"src", fmt::format("{}", item.src},
                                            //{"dst", fmt::format("{}", item.dst},
                                            {"size", std::to_string(item.size)},
                                            {"kind", std::to_string(item.kind)},
                                            };
    addEventRecord(item, profiling_start, args, event_map);
  }

  fprintf(stderr, "kernel\n");
  for (auto &item : d->kernelRows_) {
    std::initializer_list<std::pair<std::string, std::string>> args = {{"op_name", ""},
                                            //{"stream", fmt::format("{}", item.stream}},
                                            //{"kernel", demangle(hipKernelNameRef(item.function))},
                                            //{"kernel", demangle(hipKernelNameRefByPtr(item.functioniAddr, item.stream))},
                                            {"grid_x", std::to_string(item.gridX)},
                                            {"grid_y", std::to_string(item.gridY)},
                                            {"grid_z", std::to_string(item.gridZ)},
                                            {"block_x", std::to_string(item.workgroupX)},
                                            {"block_y", std::to_string(item.workgroupY)},
                                            {"block_z", std::to_string(item.workgroupZ)},
                                            };
    addEventRecord(item, profiling_start, args, event_map);
    kernelLaunches[item.id] = item;
  }
#endif

  // FIXME: remove once apis enabled
  for (auto &item : d->kernelRows_) {
    kernelLaunches[item.id] = &item;
  }
  for (auto &item : d->copyRows_) {
    copyLaunches[item.id] = &item;
  }

  for (auto& buffer : *d->gpuTraceBuffers_) {
    const roctracer_record_t* record = (const roctracer_record_t*)(buffer.data);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(buffer.data + buffer.validSize);
    while (record < end_record) {
      std::unordered_map<std::string, std::string> args;
      std::string name = roctracer_op_string(record->domain, record->op, record->kind);

      // Add kernel args if we have them
      auto kit = kernelLaunches.find(record->correlation_id);
      if (kit != kernelLaunches.end()) {
        auto &item = *(*kit).second;
        args["stream"] = "FIXME: format";
        args["grid_x"] = std::to_string(item.gridX); 
        args["grid_y"] = std::to_string(item.gridY); 
        args["grid_z"] = std::to_string(item.gridZ); 
        args["block_x"] = std::to_string(item.workgroupX); 
        args["block_y"] = std::to_string(item.workgroupY); 
        args["block_z"] = std::to_string(item.workgroupZ); 
        if (item.functionAddr != nullptr) {
          //name = demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream));
          name = hipKernelNameRefByPtr(item.functionAddr, item.stream);
        }
        else if (item.function != nullptr) {
          //name = demangle(hipKernelNameRef(item.function));
          name = hipKernelNameRef(item.function);
        }
      }

      // Add copy args if we have them
      auto cit = copyLaunches.find(record->correlation_id);
      if (cit != copyLaunches.end()) {
        auto &item = *(*cit).second;
        args["stream"] = "FIXME: format";
        args["dst"] = "FIXME: format";
        args["src"] = "FIXME: format";
        args["size"] = std::to_string(item.size);
        args["kind"] = std::to_string(item.kind);
      }

      EventRecord event{
          onnxruntime::profiling::KERNEL_EVENT,
          static_cast<int>(record->device_id),
          static_cast<int>(record->queue_id),
          name,
          static_cast<int64_t>((record->begin_ns - profiling_start) / 1000),
          static_cast<int64_t>((record->end_ns - record->begin_ns) / 1000),
          std::move(args)};

       // FIXME: deal with missing ext correlation
       auto extId = d->externalCorrelations_[RoctracerLogger::CorrelationDomain::Default][record->correlation_id];
       if (event_map.find(extId) == event_map.end()) {
         event_map.insert({extId, {event}});
       }
       else {
         event_map[extId].push_back(std::move(event));
       }

       roctracer_next_record(record, &record);
       //fprintf(stderr, "EventRecord: %s %lld (duration %lld)  ext %ld\n", event.name.c_str(), event.ts, event.dur, extId);
    }
  }

  // General 
  auto insert_iter = events.begin();
  for (auto& map_iter : event_map) {
    auto ts = static_cast<long long>(map_iter.first);
    while (insert_iter != events.end() && insert_iter->ts < ts) {
      insert_iter++;
    }
    if (insert_iter != events.end() && insert_iter->ts == ts) {
      for (auto& evt_iter : map_iter.second) {
        evt_iter.args["op_name"] = insert_iter->args["op_name"];
      }
      insert_iter = events.insert(insert_iter+1, map_iter.second.begin(), map_iter.second.end());
    } else {
      insert_iter = events.insert(insert_iter, map_iter.second.begin(), map_iter.second.end());
    }
    while (insert_iter != events.end() && insert_iter->cat == EventCategory::KERNEL_EVENT) {
      insert_iter++;
    }
  }
}

void RocmProfiler::Start(uint64_t id)
{
    d->pushCorrelationID(id, RoctracerLogger::CorrelationDomain::Default);
}

void RocmProfiler::Stop(uint64_t)
{
    d->popCorrelationID(RoctracerLogger::CorrelationDomain::Default);
}


void RocmProfiler::addEventRecord(const roctracerRow &item, int64_t pstart, const std::initializer_list<std::pair<std::string, std::string>> &args, std::map<uint64_t, std::vector<EventRecord>> &event_map)
{
  EventRecord event{onnxruntime::profiling::KERNEL_EVENT,
      static_cast<int>(item.pid),
      static_cast<int>(item.tid),
      std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0)),
      static_cast<int64_t>((item.begin - pstart) / 1000),
      static_cast<int64_t>((item.end - item.begin) / 1000),
      {args.begin(), args.end()}};

  // FIXME: deal with missing ext correlation
  auto extId = d->externalCorrelations_[RoctracerLogger::CorrelationDomain::Default][item.id];
  if (event_map.find(extId) == event_map.end()) {
    event_map.insert({extId, {event}});
  }
  else {
    event_map[extId].push_back(std::move(event));
  }
  //fprintf(stderr, "EventRecord: %s %lld (duration %lld)  ext %ld\n", event.name.c_str(), event.ts, event.dur, extId);
}

}  // namespace profiling
}  // namespace onnxruntime
#endif
