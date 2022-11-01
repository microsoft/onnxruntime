#include <time.h>
#include <cstring>
#include <chrono>

#include "roctracer_manager.h"

namespace onnxruntime {
namespace profiling {

// allocate a 16K buffer for recording async activities
static constexpr size_t sc_activity_buffer_size = 0x4000;

const std::vector<std::string> RoctracerManager::hip_api_calls_to_trace = {
  "hipMemcpy",
  "hipMemcpy2D",
  "hipMemcpyAsync",
  "hipMemcpy2DAsync",
  "hipMemcpyWithStream"
  "hipLaunchKernel",
  "hipMemset",
  "hipMemsetAsync",
  "hipExtModuleLaunchKernel"
};

// Implementation of RoctracerActivityBuffer
RoctracerActivityBuffer& RoctracerActivityBuffer::operator = (const RoctracerActivityBuffer& other) {
  if (&other == this) {
    return *this;
  }
  if (data_ != nullptr) {
    free(data_);
  }

  size_ = other.size_;
  data_ = (uint8_t*)malloc(size_);
  memcpy(data_, other.data_, size_);
  return *this;
}

RoctracerActivityBuffer& RoctracerActivityBuffer::operator = (RoctracerActivityBuffer&& other) {
  if (&other == this) {
    return *this;
  }
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
  return *this;
}

RoctracerActivityBuffer::~RoctracerActivityBuffer() {
  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
  }
}

// Implementation of RoctracerManager
RoctracerManager& RoctracerManager::GetInstance() {
  static RoctracerManager instance;
  return instance;
}

RoctracerManager::~RoctracerManager() {
  StopLogging();
}

uint64_t RoctracerManager::RegisterClient() {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  auto res = next_client_id_++;
  per_client_events_by_ext_correlation_.insert({res, {}});
  return res;
}

void RoctracerManager::DeregisterClient(uint64_t client_handle) {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  per_client_events_by_ext_correlation_.erase(client_handle);
}

void RoctracerManager::StartLogging() {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  if (logging_enabled_) {
    return;
  }

  // The following line shows up in all the samples, I do not know
  // what the point is, but without it, the roctracer APIs don't work.
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);

  roctracer_properties_t hcc_cb_properties;
  memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
  hcc_cb_properties.buffer_size = sc_activity_buffer_size;
  hcc_cb_properties.buffer_callback_fun = ActivityCallback;
  roctracer_open_pool(&hcc_cb_properties);

  // Enable selective activity and API callbacks for the HIP APIs
  roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);

  for (auto const& logged_api : hip_api_calls_to_trace) {
    uint32_t cid = 0;
    roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, logged_api.c_str(), &cid, nullptr);
    roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, cid, ApiCallback, nullptr);
    roctracer_enable_op_activity(ACTIVITY_DOMAIN_HIP_API, cid);
  }

  // Enable activity logging in the HIP_OPS/HCC_OPS domain.
  roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);

  roctracer_start();
  logging_enabled_ = true;
}

void RoctracerManager::StopLogging() {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  if (!logging_enabled_) {
    return;
  }

  roctracer_stop();
  roctracer_flush_activity();
  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
  roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer_close_pool();
  roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
  logging_enabled_ = false;
}

void RoctracerManager::Consume(uint64_t client_handle, const TimePoint& start_time,
                               std::map<uint64_t, Events>& events) {
  events.clear();
  std::vector<RoctracerActivityBuffer> activity_buffers;
  {
    std::lock_guard<std::mutex> lock(unprocessed_activity_buffers_lock_);
    std::swap(unprocessed_activity_buffers_, activity_buffers);
    unprocessed_activity_buffers_.clear();
  }

  {
    // Ensure that at most one thread is working through the activity buffers at any time.
    std::lock_guard<std::mutex> lock_two(activity_buffer_processor_mutex_);
    ProcessActivityBuffers(activity_buffers, start_time);
    std::lock_guard<std::mutex> lock(event_list_mutex_);
    auto it = per_client_events_by_ext_correlation_.find(client_handle);
    if (it == per_client_events_by_ext_correlation_.end()) {
      return;
    }
    std::swap(events, it->second);
  }
}

bool RoctracerManager::PushCorrelation(uint64_t client_handle, uint64_t external_correlation_id) {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  auto it = per_client_events_by_ext_correlation_.find(client_handle);
  if (it == per_client_events_by_ext_correlation_.end()) {
    return false;
  }
  roctracer_activity_push_external_correlation_id(external_correlation_id);
  // assumption: correlation_ids are unique. This should generally work,
  // because we use timestamps as correlation ids.
  external_correlation_id_to_client_[external_correlation_id] = client_handle;
  return true;
}

void RoctracerManager::PopCorrelation(uint64_t& popped_external_correlation_id) {
  std::lock_guard<std::mutex> lock(roctracer_manager_mutex_);
  roctracer_activity_pop_external_correlation_id(&popped_external_correlation_id);
}

void RoctracerManager::ActivityCallback(const char* begin, const char* end, void* arg) {
  size_t size = end - begin;
  RoctracerActivityBuffer activity_buffer{reinterpret_cast<const uint8_t*>(begin), size};
  auto& instance = GetInstance();
  {
    std::lock_guard<std::mutex> lock(instance.unprocessed_activity_buffers_lock_);
    instance.unprocessed_activity_buffers_.emplace_back(std::move(activity_buffer));
  }
}

void RoctracerManager::ApiCallback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  if (domain != ACTIVITY_DOMAIN_HIP_API) {
    return;
  }
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  if (data->phase == ACTIVITY_API_PHASE_EXIT) {
    // We only save args for async launches on the ACTIVITY_API_PHASE_ENTER phase
    return;
  }

  auto& instance = GetInstance();
  {
    std::lock_guard<std::mutex> lock(instance.api_call_args_lock_);
    auto& record = instance.api_call_args_[data->correlation_id];
    record.domain_ = domain;
    record.cid_ = cid;
    record.api_data_ = *data;
  }
}

static inline std::string PointerToHexString(const void* ptr) {
  std::ostringstream sstr;
  sstr << std::hex << ptr;
  return sstr.str();
}

void RoctracerManager::CreateEventForKernelRecord(const roctracer_record_t* record,
                                                  uint64_t start_time_ns,
                                                  const ApiCallRecord& call_record,
                                                  EventRecord& event) {
  auto const& launch_args = call_record.api_data_;
  auto name = demangle(hipKernelNameRefByPtr(launch_args.args.hipLaunchKernel.function_address,
                                             launch_args.args.hipLaunchKernel.stream));
  std::unordered_map<std::string, std::string> args {
    {"stream", PointerToHexString((void*)(launch_args.args.hipLaunchKernel.stream))},
    {"grid_x", std::to_string(launch_args.args.hipLaunchKernel.numBlocks.x)},
    {"grid_y", std::to_string(launch_args.args.hipLaunchKernel.numBlocks.y)},
    {"grid_z", std::to_string(launch_args.args.hipLaunchKernel.numBlocks.z)},
    {"block_x", std::to_string(launch_args.args.hipLaunchKernel.dimBlocks.x)},
    {"block_y", std::to_string(launch_args.args.hipLaunchKernel.dimBlocks.y)},
    {"block_z", std::to_string(launch_args.args.hipLaunchKernel.dimBlocks.z)}
  };

  new (&event) EventRecord {
    /* cat = */ EventCategory::KERNEL_EVENT,
    /* pid = */ -1,
    /* tid = */ -1,
    /* name = */ std::move(name),
    /* ts = */ (int64_t)(record->begin_ns - start_time_ns) / 1000,
    /* dur = */ (int64_t)(record->end_ns - record->begin_ns) / 1000,
    /* args = */ std::move(args)
  };
}

void RoctracerManager::CreateEventForMemsetRecord(const roctracer_record_t* record,
                                                  uint64_t start_time_ns,
                                                  const ApiCallRecord& call_record,
                                                  EventRecord& event) {
  auto const& launch_args = call_record.api_data_;
  auto dst_string = PointerToHexString(launch_args.args.hipMemset.dst);
  std::string name {roctracer_op_string(call_record.domain_, call_record.cid_, 0)};

  std::unordered_map<std::string, std::string> args {
    {"stream", call_record.cid_ == HIP_API_ID_hipMemset
                                ? "0"
                                : PointerToHexString((void*)launch_args.args.hipMemsetAsync.stream)},
    {"dst", dst_string},
    {"size", std::to_string(launch_args.args.hipMemset.sizeBytes)},
    {"value", std::to_string(launch_args.args.hipMemset.value)}
  };
  new (&event) EventRecord {
    /* cat = */ EventCategory::KERNEL_EVENT,
    /* pid = */ -1,
    /* tid = */ -1,
    /* name = */ std::move(name),
    /* ts = */ (int64_t)(record->begin_ns - start_time_ns) / 1000,
    /* dur = */ (int64_t)(record->end_ns - record->begin_ns) / 1000,
    /* args = */ std::move(args)
  };
}

static inline std::string MemcpyKindToString(hipMemcpyKind kind) {
  switch(kind) {
    case hipMemcpyHostToHost:
      return "H2H";
    case hipMemcpyHostToDevice:
      return "H2D";
    case hipMemcpyDeviceToHost:
      return "D2H";
    case hipMemcpyDeviceToDevice:
      return "D2D";
    default:
      return "Default";
  }
}

void RoctracerManager::CreateEventForMemcpyRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                                  const ApiCallRecord& call_record, EventRecord& event) {
  auto const& launch_args = call_record.api_data_;
  auto src_string = PointerToHexString(launch_args.args.hipMemcpy.src);
  auto dst_string = PointerToHexString(launch_args.args.hipMemcpy.dst);
  std::string name {roctracer_op_string(call_record.domain_, call_record.cid_, 0)};

  std::string memcpy_kind_string = MemcpyKindToString(launch_args.args.hipMemcpy.kind);

  std::unordered_map<std::string, std::string> args {
    {"stream", call_record.cid_ == HIP_API_ID_hipMemcpy
                                ? "0"
                                : PointerToHexString((void*)launch_args.args.hipMemcpyAsync.stream)},
    {"src", src_string},
    {"dst", dst_string},
    {"kind", memcpy_kind_string}
  };

  new (&event) EventRecord {
    /* cat = */ EventCategory::KERNEL_EVENT,
    /* pid = */ -1,
    /* tid = */ -1,
    /* name = */ std::move(name),
    /* ts = */ (int64_t)(record->begin_ns - start_time_ns) / 1000,
    /* dur = */ (int64_t)(record->end_ns - record->begin_ns) / 1000,
    /* args = */ std::move(args)
  };
}

void RoctracerManager::CreateEventForMemcpy2DRecord(const roctracer_record_t* record, uint64_t start_time_ns,
                                                    const ApiCallRecord& call_record, EventRecord& event) {
  auto const& launch_args = call_record.api_data_;
  auto src_string = PointerToHexString(launch_args.args.hipMemcpy2D.src);
  auto dst_string = PointerToHexString(launch_args.args.hipMemcpy2D.dst);
  std::string name {roctracer_op_string(call_record.domain_, call_record.cid_, 0)};

  std::string memcpy_kind_string = MemcpyKindToString(launch_args.args.hipMemcpy2D.kind);

  std::unordered_map<std::string, std::string> args {
    {"stream", call_record.cid_ == HIP_API_ID_hipMemcpy2D
                                ? "0"
                                : PointerToHexString((void*)launch_args.args.hipMemcpy2DAsync.stream)},
    {"src", src_string},
    {"dst", dst_string},
    {"spitch", std::to_string(launch_args.args.hipMemcpy2D.spitch)},
    {"dpitch", std::to_string(launch_args.args.hipMemcpy2D.dpitch)},
    {"width", std::to_string(launch_args.args.hipMemcpy2D.width)},
    {"height", std::to_string(launch_args.args.hipMemcpy2D.height)},
    {"kind", memcpy_kind_string}
  };

  new (&event) EventRecord {
    /* cat = */ EventCategory::KERNEL_EVENT,
    /* pid = */ -1,
    /* tid = */ -1,
    /* name = */ std::move(name),
    /* ts = */ (int64_t)(record->begin_ns - start_time_ns) / 1000,
    /* dur = */ (int64_t)(record->end_ns - record->begin_ns) / 1000,
    /* args = */ std::move(args)
  };
}

void RoctracerManager::CreateEventForExtModuleLaunchKernel(const roctracer_record_t* record, uint64_t start_time_ns,
                                                           const ApiCallRecord& call_record, EventRecord& event) {
  auto const& launch_args = call_record.api_data_;
  auto name = hipKernelNameRef(launch_args.args.hipExtModuleLaunchKernel.f);
  std::unordered_map<std::string, std::string> args {
    {"stream", PointerToHexString((void*)launch_args.args.hipExtModuleLaunchKernel.hStream)},
    {"grid_x", std::to_string(launch_args.args.hipExtModuleLaunchKernel.globalWorkSizeX)},
    {"grid_y", std::to_string(launch_args.args.hipExtModuleLaunchKernel.globalWorkSizeY)},
    {"grid_z", std::to_string(launch_args.args.hipExtModuleLaunchKernel.globalWorkSizeZ)},
    {"block_x", std::to_string(launch_args.args.hipExtModuleLaunchKernel.localWorkSizeX)},
    {"block_y", std::to_string(launch_args.args.hipExtModuleLaunchKernel.localWorkSizeY)},
    {"block_z", std::to_string(launch_args.args.hipExtModuleLaunchKernel.localWorkSizeZ)},
  };

  new (&event) EventRecord {
    /* cat = */ EventCategory::KERNEL_EVENT,
    /* pid = */ -1,
    /* tid = */ -1,
    /* name = */ std::move(name),
    /* ts = */ (int64_t)(record->begin_ns - start_time_ns) / 1000,
    /* dur = */ (int64_t)(record->end_ns - record->begin_ns) / 1000,
    /* args = */ std::move(args)
  };
}

void RoctracerManager::MapEventsToClient(uint64_t external_correlation_id, Events&& events) {
  auto client_it = external_correlation_id_to_client_.find(external_correlation_id);
  if (client_it == external_correlation_id_to_client_.end()) {
    // :-( well, we tried really, really hard to map this event to a client.
    return;
  }
  auto& event_list = per_client_events_by_ext_correlation_[client_it->second][client_it->first];
  event_list.insert(event_list.end(),
                    std::make_move_iterator(events.begin()),
                    std::make_move_iterator(events.end()));
}

void RoctracerManager::MapEventToClient(uint64_t external_correlation_id, EventRecord&& event) {
  auto client_it = external_correlation_id_to_client_.find(external_correlation_id);
  if (client_it == external_correlation_id_to_client_.end()) {
    // :-( well, we tried really, really hard to map this event to a client.
    return;
  }
  per_client_events_by_ext_correlation_[client_it->second][client_it->first].emplace_back(std::move(event));
}

void RoctracerManager::ProcessActivityBuffers(const std::vector<RoctracerActivityBuffer>& buffers,
                                              const TimePoint& start_time) {
  std::unordered_map<uint64_t, std::vector<EventRecord>> events_pending_client_mapping;
  auto start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time.time_since_epoch()).count();

  for (auto const& buffer : buffers) {
    auto current_record = reinterpret_cast<const roctracer_record_t*>(buffer.GetData());
    auto data_end = reinterpret_cast<const roctracer_record_t*>(buffer.GetData() + buffer.GetSize());
    for ( ; current_record < data_end; roctracer_next_record(current_record, &current_record)) {
      EventRecord event;
      if (current_record->domain == ACTIVITY_DOMAIN_EXT_API) {
        roctracer_correlation_to_external_correlation_[current_record->correlation_id] = current_record->external_id;

        // check for any events pending client mapping on this correlation
        auto pending_it = events_pending_client_mapping.find(current_record->correlation_id);
        if (pending_it == events_pending_client_mapping.end()) {
          continue;
        }

        // we have one or more pending events, map them to the client
        MapEventsToClient(current_record->external_id, std::move(pending_it->second));
        events_pending_client_mapping.erase(current_record->correlation_id);

      } else if (current_record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
        if (current_record->op == 1 && current_record->kind == HipOpMarker) {
          // this is just a marker, ignore it.
          continue;
        }

        auto api_it = api_call_args_.find(current_record->correlation_id);
        if (api_it == api_call_args_.end()) {
            // we're not tracking this activity, ignore it
          continue;
        }

        auto const& call_record = api_it->second;
        switch (call_record.cid_)
        {
        case HIP_API_ID_hipLaunchKernel:
          CreateEventForKernelRecord(current_record, start_time_ns, call_record, event);
          break;

        case HIP_API_ID_hipMemset:
        case HIP_API_ID_hipMemsetAsync:
          CreateEventForMemsetRecord(current_record, start_time_ns, call_record, event);
          break;

        case HIP_API_ID_hipMemcpy:
        case HIP_API_ID_hipMemcpyAsync:
        case HIP_API_ID_hipMemcpyWithStream:
          CreateEventForMemcpyRecord(current_record, start_time_ns, call_record, event);
          break;

        case HIP_API_ID_hipMemcpy2D:
        case HIP_API_ID_hipMemcpy2DAsync:
          CreateEventForMemcpy2DRecord(current_record, start_time_ns, call_record, event);
          break;

        case HIP_API_ID_hipExtModuleLaunchKernel:
          CreateEventForExtModuleLaunchKernel(current_record, start_time_ns, call_record, event);
          break;

        default:
          break;
        }
      }

      // map the event to the right client
      auto ext_corr_it = roctracer_correlation_to_external_correlation_.find(current_record->correlation_id);
      if (ext_corr_it == roctracer_correlation_to_external_correlation_.end()) {
        // defer the processing of this event
        events_pending_client_mapping[current_record->correlation_id].emplace_back(std::move(event));
        continue;
      }
      MapEventToClient(ext_corr_it->second, std::move(event));
    }
  }
}

} /* end namespace profiling */
} /* end namespace onnxruntime */


/*
typedef uint64_t timestamp_t;

static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
}

//using namespace std::chrono;

RoctracerLogger& RoctracerLogger::singleton() {
  static RoctracerLogger instance;
  return instance;
}

RoctracerLogger::RoctracerLogger() {
  gpuTraceBuffers_ = std::make_unique<std::list<RoctracerActivityBuffer>>();
}

RoctracerLogger::~RoctracerLogger() {
  stopLogging();
  endTracing();
}

namespace {
  thread_local std::deque<uint64_t> t_externalIds[RoctracerLogger::CorrelationDomain::size];
}

void RoctracerLogger::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  t_externalIds[type].push_back(id);
}

void RoctracerLogger::popCorrelationID(CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  t_externalIds[type].pop_back();
}

void RoctracerLogger::clearLogs() {
  rows_.clear();
  kernelRows_.clear();
  copyRows_.clear();
  mallocRows_.clear();
  gpuTraceBuffers_->clear();
  for (int i = 0; i < CorrelationDomain::size; ++i) {
    externalCorrelations_[i].clear();
  }
}

void RoctracerLogger::api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  RoctracerLogger *dis = &singleton();

  if (domain == ACTIVITY_DOMAIN_HIP_API && dis->loggedIds_.contains(cid)) {
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);

    // Pack callbacks into row structures

    thread_local timespec timestamp;

    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      clock_gettime(CLOCK_MONOTONIC, &timestamp);  // record proper clock
    } else {  // (data->phase == ACTIVITY_API_PHASE_EXIT)
      timespec endTime;
      timespec startTime { timestamp };
      clock_gettime(CLOCK_MONOTONIC, &endTime);  // record proper clock

      switch (cid) {
        case HIP_API_ID_hipLaunchKernel:
        case HIP_API_ID_hipExtLaunchKernel:
        case HIP_API_ID_hipLaunchCooperativeKernel:     // Should work here
          {
          auto &args = data->args.hipLaunchKernel;
          dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.function_address,
                              nullptr,
                              args.numBlocks.x,
                              args.numBlocks.y,
                              args.numBlocks.z,
                              args.dimBlocks.x,
                              args.dimBlocks.y,
                              args.dimBlocks.z,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
          break;
        case HIP_API_ID_hipHccModuleLaunchKernel:
        case HIP_API_ID_hipModuleLaunchKernel:
        case HIP_API_ID_hipExtModuleLaunchKernel:
          {
          auto &args = data->args.hipModuleLaunchKernel;
          dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              nullptr,
                              args.f,
                              args.gridDimX,
                              args.gridDimY,
                              args.gridDimZ,
                              args.blockDimX,
                              args.blockDimY,
                              args.blockDimZ,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
          break;
        case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
        case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
#if 0
          {
            auto &args = data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
            dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.function_address,
                              nullptr,
                              args.numBlocks.x,
                              args.numBlocks.y,
                              args.numBlocks.z,
                              args.dimBlocks.x,
                              args.dimBlocks.y,
                              args.dimBlocks.z,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
#endif
          break;
        case HIP_API_ID_hipMalloc:
            dis->mallocRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipMalloc.ptr__val,
                              data->args.hipMalloc.size
                              );
          break;
        case HIP_API_ID_hipFree:
            dis->mallocRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipFree.ptr,
                              0
                              );
          break;
        case HIP_API_ID_hipMemcpy:
          {
            auto &args = data->args.hipMemcpy;
            dis->copyRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.src,
                              args.dst,
                              args.sizeBytes,
                              args.kind,
                              static_cast<hipStream_t>(0)  // use placeholder?
                              );
          }
          break;
        case HIP_API_ID_hipMemcpyAsync:
        case HIP_API_ID_hipMemcpyWithStream:
          {
            auto &args = data->args.hipMemcpyAsync;
            dis->copyRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.src,
                              args.dst,
                              args.sizeBytes,
                              args.kind,
                              args.stream
                              );
          }
          break;
        default:
          dis->rows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime)
                              );
          break;
      }  // switch
      // External correlation
      for (int it = CorrelationDomain::begin; it < CorrelationDomain::end; ++it) {
        if (t_externalIds[it].size() > 0) {
          dis->externalCorrelations_[it][data->correlation_id] = t_externalIds[it].back();
        }
      }
    }  // phase exit
  }
}

void RoctracerLogger::activity_callback(const char* begin, const char* end, void* arg)
{
  size_t size = end - begin;
  uint8_t *buffer = (uint8_t*) malloc(size);
  auto &gpuTraceBuffers = singleton().gpuTraceBuffers_;
  memcpy(buffer, begin, size);
  gpuTraceBuffers->emplace_back(buffer, size);
}

void RoctracerLogger::startLogging() {
  if (!registered_) {
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);  // Magic encantation

    // Set some api calls to ignore
    loggedIds_.setInvertMode(true);  // Omit the specified api
    loggedIds_.add("hipGetDevice");
    loggedIds_.add("hipSetDevice");
    loggedIds_.add("hipGetLastError");
    loggedIds_.add("__hipPushCallConfiguration");
    loggedIds_.add("__hipPopCallConfiguration");
    loggedIds_.add("hipCtxSetCurrent");
    loggedIds_.add("hipEventRecord");
    loggedIds_.add("hipEventQuery");
    loggedIds_.add("hipGetDeviceProperties");
    loggedIds_.add("hipPeekAtLastError");
    loggedIds_.add("hipModuleGetFunction");
    loggedIds_.add("hipEventCreateWithFlags");

    // Enable API callbacks
    if (loggedIds_.invertMode() == true) {
        // exclusion list - enable entire domain and turn off things in list
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, nullptr);
        const std::unordered_map<uint32_t, uint32_t> &filter = loggedIds_.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first);
        }
    }
    else {
        // inclusion list - only enable things in the list
        const std::unordered_map<uint32_t, uint32_t> &filter = loggedIds_.filterList();
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first, api_callback, nullptr);
        }
    }
    //roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, nullptr);

    // Allocate default tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    roctracer_open_pool(&properties);

    // Enable async op collection
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    hcc_cb_properties.buffer_size = 0x4000;
    hcc_cb_properties.buffer_callback_fun = activity_callback;
    roctracer_open_pool_expl(&hcc_cb_properties, &hccPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hccPool_);

    registered_ = true;
  }

  externalCorrelationEnabled_ = true;
  roctracer_start();
}

void RoctracerLogger::stopLogging() {
  roctracer_stop();
  roctracer_flush_activity_expl(hccPool_);
}

void RoctracerLogger::endTracing() {
  if (registered_ == true) {
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    //roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
    roctracer_close_pool_expl(hccPool_);
  }
}


ApiIdList::ApiIdList()
: invert_(true)
{
}

void ApiIdList::add(const std::string &apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) == ROCTRACER_STATUS_SUCCESS) {
    filter_[cid] = 1;
  }
}
void ApiIdList::remove(const std::string &apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) == ROCTRACER_STATUS_SUCCESS) {
    filter_.erase(cid);
  }
}

bool ApiIdList::loadUserPrefs()
{
  // placeholder
  return false;
}
bool ApiIdList::contains(uint32_t apiId)
{
  return (filter_.find(apiId) != filter_.end()) ? !invert_ : invert_;  // XOR
}
*/
